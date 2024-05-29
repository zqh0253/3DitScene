from dataclasses import dataclass, field
import pytorch_lightning as pl
from threestudio.utils.config import parse_structured
from threestudio.utils.base import Updateable, update_if_possible
from threestudio.utils.saving import SaverMixin
from threestudio.utils.typing import *

import open_clip
import torch
import torchvision
from torch import nn
import cv2
import numpy as np
from sklearn.decomposition import PCA

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from mobile_sam import sam_model_registry as m_sam_model_registry
from mobile_sam import SamAutomaticMaskGenerator as m_SamAutomaticMaskGenerator
from mobile_sam import SamPredictor as m_SamPredictor

@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims

    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)

def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def sava_numpy(save_path, data):
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, data['seg_maps'].numpy())
    np.save(save_path_f, data['feature'].numpy())

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]

    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr

    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def sam_encoder(image, mask_generator):
    image = image.detach().cpu()
    image = cv2.cvtColor(image[0].permute(1,2,0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    # pre-compute masks
    masks_l = mask_generator.generate(image)
    # pre-compute postprocess
    masks_l = masks_update(masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)[0]

    def mask2segmap(masks, image):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            mask = masks[i]
            seg_img = get_seg_img(mask, image)
            pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
            seg_img_list.append(pad_seg_img)

            seg_map[masks[i]['segmentation']] = i
        seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')

        return seg_imgs, seg_map

    seg_images, seg_maps = {}, {}
    seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image)

    # 0:default 1:s 2:m 3:l
    return seg_images, seg_maps

class SamClip(pl.LightningModule, Updateable, SaverMixin):
    @dataclass
    class Config:
        clip_model_type: str = "ViT-B-16"
        clip_model_pretrained: str = "laion2b_s34b_b88k"
        clip_n_dims: int = 512
        sam_ckpt_path: str = "ckpts/sam_vit_h_4b8939.pth"
        feature_level: int = 3
        vis_pca_feature: bool = True
        use_mobile_sam: bool = True

    cfg: Config

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self.model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
        self.clip_n_dims = self.cfg.clip_n_dims
        self.tokenizer = open_clip.get_tokenizer(self.cfg.clip_model_type)
        sam = sam_model_registry["vit_h"](checkpoint=self.cfg.sam_ckpt_path).to('cuda')
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.7,
            box_nms_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
        )

        model_type = "vit_t"
        sam_checkpoint = "./ckpts/mobile_sam.pt"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mobile_sam = m_sam_model_registry[model_type](checkpoint=sam_checkpoint)
        mobile_sam.to(device=device)
        mobile_sam.eval()
        self.m_mask_generator = m_SamAutomaticMaskGenerator(mobile_sam)

        # self.estimator = PCA(n_components=3)
        # self.has_fit = False

        self.mask_generator.predictor.model.to('cuda')
        self.m_mask_generator.predictor.model.to('cuda')
        
    def _embed_clip_sam_tiles(self, image, sam_encoder):
        aug_imgs = torch.cat([image])
        if self.cfg.use_mobile_sam:
            seg_images, seg_map = sam_encoder(aug_imgs, self.m_mask_generator)
        else:
            seg_images, seg_map = sam_encoder(aug_imgs, self.mask_generator)

        clip_embeds = {}
        # types = ['default', 's', 'm', 'l']
        types = ['l']
        for mode in types:
            tiles = seg_images[mode]
            tiles = tiles.to("cuda")
            with torch.no_grad():
                clip_embed = self.model.encode_image(tiles)
            clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
            clip_embeds[mode] = clip_embed.detach().cpu().half()
        
        return clip_embeds, seg_map

    def forward(self, img):
        embed_size=512
        seg_maps = []
        total_lengths = []
        timer = 0
        img_embeds = torch.zeros((len(img), 100, embed_size))
        
        seg_maps = torch.zeros((len(img), 1, *img.shape[2:]))
        img_embed, seg_map = self._embed_clip_sam_tiles(img, sam_encoder)

        lengths = [len(v) for k, v in img_embed.items()]
        total_length = sum(lengths)
        # total_lengths.append(total_length)

        # if total_length > img_embeds.shape[1]:
        #     pad = total_length - img_embeds.shape[1]
        #     img_embeds = torch.cat([
        #         img_embeds,
        #         torch.zeros((len(image_list), pad, embed_size))
        #     ], dim=1)

        # img_embed = torch.cat([v for k, v in img_embed.items()], dim=0)
        # assert img_embed.shape[0] == total_length
        img_embeds[0, :total_length] = img_embed['l']

        # seg_map_tensor = []
        # lengths_cumsum = lengths.copy()
        # for j in range(1, len(lengths)):
        #     lengths_cumsum[j] += lengths_cumsum[j-1]
        # for j, (k, v) in enumerate(seg_map.items()):
        #     if j == 0:
        #         seg_map_tensor.append(torch.from_numpy(v))
        #         continue
        #     assert v.max() == lengths[j] - 1, f"{j}, {v.max()}, {lengths[j]-1}"
        #     v[v != -1] += lengths_cumsum[j-1]
        #     seg_map_tensor.append(torch.from_numpy(v))
        # seg_map = torch.stack(seg_map_tensor, dim=0)
        seg_maps[0] = torch.from_numpy(seg_map['l'])

        # self.mask_generator.predictor.model.to('cpu')
        feature_map = img_embeds[0]   # 300, 512
        seg_map = seg_maps[0]    # 4, 512, 512

        image_height, image_width = seg_map.shape[1:]
        y, x = torch.meshgrid(torch.arange(0, image_height), torch.arange(0, image_width))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        seg = seg_map[:, y, x].squeeze(-1).long()
        mask = seg != -1
        point_feature1 = feature_map[seg[:]].squeeze(0)
        mask = mask[:].reshape(1, image_height, image_width)
        return img_embed['l'], seg, mask
        # point_feature = point_feature1.reshape(image_height, image_width, -1).permute(2, 0, 1)

        # return img_embed['l'], point_feature, mask
        
