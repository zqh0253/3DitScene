import math
from dataclasses import dataclass, field

import os
import collections   
import random
import numpy as np
import threestudio
import torch
import cv2
from sklearn.cluster import KMeans
import torchvision
from PIL import Image
from transformers import pipeline
from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.loss import tv_loss
from threestudio.utils.ops import get_cam_info_gaussian
from threestudio.utils.typing import *
from torch.cuda.amp import autocast
from tqdm.contrib import tenumerate  
from tqdm import tqdm, trange

from ..geometry.gaussian_base import BasicPointCloud, Camera
from ..utils.sam_clip import SamClip
from ..utils.ae import Autoencoder_dataset, Autoencoder
from torch.utils.data import Dataset, DataLoader

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def cos_loss(network_output, gt):
    return 1 - torch.nn.functional.cosine_similarity(network_output, gt, dim=0).mean()


@threestudio.register("scene-lang-system")
class SceneLang(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False

        distill_lang_freq: int = 800
        outpaint_step: int = 300
        sam_clip: dict = field(default_factory=dict)
        encoder_hidden_dims: Optional[List] = field(default_factory=list) 
        decoder_hidden_dims: Optional[List] = field(default_factory=list) 
        ae_epoch: int = 100
        distill_lang_epoch: int = 100
        sam_clip_ae_lr: float = 3e-4
        densify: bool = True
        distill_interval: int = 2
        xyz_noise_ratio: Any = None
        drop_ooi_ratio: Any = field(default_factory=dict)
        empty_prompt: str = "empty"
        side_prompt: str = "empty"
        crop_with_lang: bool = True
        rotate_aug_scale: int = 15

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.automatic_optimization = False

        self.geometry.prompt = self.cfg.prompt_processor.prompt
        self.geometry.empty_prompt = self.cfg.empty_prompt
        self.geometry.side_prompt = self.cfg.side_prompt

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()

        self.cfg.prompt_processor.prompt = self.cfg.empty_prompt
        self.bg_prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.bg_prompt_utils = self.bg_prompt_processor()

        self.sam_clip = SamClip(self.cfg.sam_clip)
        self.sam_clip_ae = Autoencoder(self.cfg.encoder_hidden_dims, self.cfg.decoder_hidden_dims).cuda()

    def configure_optimizers(self):
        optim = self.geometry.optimizer
        if hasattr(self, "merged_optimizer"):
            return [optim]
        if hasattr(self.cfg.optimizer, "name"):
            net_optim = parse_optimizer(self.cfg.optimizer, self)
            optim = self.geometry.merge_optimizer(net_optim)
            self.merged_optimizer = True
        else:
            self.merged_optimizer = False
        return [optim]

    def on_save_checkpoint(self, checkpoint):
        if 'optimizer_states' in checkpoint.keys():
            del checkpoint['optimizer_states']
        
        del_keys = [k for k in checkpoint['state_dict'].keys() if 'sam' in k]
        for k in del_keys:
            del checkpoint['state_dict'][k]
   
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.geometry.update_learning_rate(self.global_step)
        outputs = self.renderer.batch_forward(batch)
        return outputs

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def training_step(self, batch, batch_idx):
        self.geometry.noise_ratio = self.C(self.cfg.xyz_noise_ratio)
        if random.random() < self.C(self.cfg.drop_ooi_ratio):
            self.geometry._opacity_mask = (sum(self.geometry.ooi_masks)==0).float()
        else:
            self.geometry._opacity_mask = None
  
        if self.true_global_step > 0 and self.true_global_step == self.cfg.distill_lang_freq :  # finish rgb phase
            self.distill_language_feature()

        if self.true_global_step == self.cfg.outpaint_step:
            self.outpaint()

        apply_rotate = False
        if self.true_global_step > self.cfg.distill_lang_freq:
            apply_rotate = random.random() < 0.5
            self.geometry.random_rotate(self.cfg.rotate_aug_scale, apply_rotate)

        opt = self.optimizers()
        out = self(batch)

        visibility_filter = out["visibility_filter"]
        radii = out["radii"]
        guidance_inp = out["comp_rgb"]
        viewspace_point_tensor = out["viewspace_points"]
        if self.geometry._opacity_mask is None:
            pu = self.prompt_utils
        else:
            pu = self.bg_prompt_utils 
        guidance_out = self.guidance(
            guidance_inp, pu, **batch, rgb_as_latents=False
        )

        loss_sds = 0.0
        loss = 0.0

        self.log(
            "gauss_num",
            int(self.geometry.get_xyz.shape[0]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
                 
        if self.cfg.loss["lambda_ref"] > 0.0:
            ref_img = self.cfg.geometry.geometry_convert_from[len("depth:"):]
            ref_img = torch.tensor(np.array(Image.open(ref_img).resize((self.dataset.cfg.width, self.dataset.cfg.height)))[None] / 255, device = out['comp_rgb'].device)
            bg_ref_img = torch.tensor(self.geometry.bg_image[None] / 255, device = out['comp_rgb'].device)
            bg_ref_img_mask = torch.from_numpy(self.geometry.bg_image_mask[None, ..., None].astype(float)).cuda()
               
            if self.geometry._opacity_mask is None: 
                if not apply_rotate:
                    l1loss = torch.nn.L1Loss()(out['comp_rgb'][0:1], ref_img)    # only calculate the first view (zero view)
                    self.log(f"train/recon_front_view", l1loss)
                    loss += l1loss * self.cfg.loss["lambda_ref"]

                    if self.true_global_step > self.cfg.outpaint_step:
                        for view_idx in [0, -1]:
                            self.geometry._opacity_mask = None
                            sample = self.trainer.val_dataloaders.dataset[view_idx]
                            for k in sample.keys():
                                try:
                                    sample[k] = sample[k].cuda()[None]
                                except:
                                    pass
                            output = self(sample)
                            rgb = output['comp_rgb']
                            target = self.outpaint_view[view_idx]
                            # loss += torch.nn.L1Loss()(rgb, target) * self.cfg.loss["lambda_ref"] 
                            loss += (torch.nn.L1Loss(reduction='none')(rgb, target) * self.outpaint_mask[view_idx]).mean() * self.cfg.loss["lambda_ref"] 
            else:
                ratio = bg_ref_img_mask.sum() / bg_ref_img_mask.shape[1] /  bg_ref_img_mask.shape[2]
                l1loss = torch.nn.L1Loss(reduction='none')(out['comp_rgb'][0:1], bg_ref_img) * bg_ref_img_mask   # only calculate the first view (zero view)
                l1loss = l1loss.mean() / ratio
                loss += l1loss * self.cfg.loss["lambda_ref"] 

        if self.cfg.loss["lambda_scaling"] > 0.0:
            scaling_loss = self.geometry.get_scaling.mean()
            loss += scaling_loss * self.cfg.loss["lambda_scaling"]
 
        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss_sds += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_")]
                )

        loss = loss + loss_sds
        iteration = self.global_step
        opt.zero_grad()
        if loss > 0:
            loss.backward(retain_graph=True)
        if self.cfg.densify:
            self.geometry.update_states(
                iteration,
                visibility_filter,
                radii,
                viewspace_point_tensor,
            )
        opt.step()
        opt.zero_grad(set_to_none=True)

        self.log("train/loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        self.geometry._opacity_mask = None
        out = self(batch)
        mask, _ = self.geometry.project_pc(batch['c2w'], H=self.dataset.cfg.height, W=self.dataset.cfg.width)
        self.save_image_grid(
            f"it{self.global_step}-val/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="validation_step",
            step=self.global_step,
        )

    def on_validation_epoch_end(self):
        self.save_img_sequence(
            f"it{self.global_step}-val",
            f"it{self.global_step}-val",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="val",
            step=self.global_step,
            delete_images=True,
        )

    def test_step(self, batch, batch_idx):
        # remove the random rotation effect!
        self.geometry.recover_xyzrot()            
        out = self(batch)
        self.save_image_grid(
            f"it{self.global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": out["lang"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (out["lang"][0].min().item(), out["lang"][0].max().item())},
                },
            ],
            name="test_step",
            step=self.global_step,
        )
        if batch["index"][0] == 0:
            save_path = self.get_save_path("point_cloud.ply")
            self.geometry.save_ply(save_path)

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.global_step}-test",
            f"it{self.global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.global_step,
        )

    def on_load_checkpoint(self, ckpt_dict) -> None:
        for key in self.state_dict().keys():
            if 'sam' in key:
                ckpt_dict["state_dict"][key] = self.state_dict()[key]

        num_pts = ckpt_dict["state_dict"]["geometry._xyz"].shape[0]
        pcd = BasicPointCloud(
            points=np.zeros((num_pts, 3)),
            colors=np.zeros((num_pts, 3)),
            normals=np.zeros((num_pts, 3)),
        )
        self.geometry.create_from_pcd(pcd, 10)
        self.geometry.training_setup()
        super().on_load_checkpoint(ckpt_dict)

    def outpaint(self) -> None:
        threestudio.info("Start outpainting.")
        self.outpaint_view = dict()
        self.outpaint_mask = dict()
        cnt = 0
        for view_idx in [0, -1]:
            self.geometry._opacity_mask = None
            sample = self.trainer.val_dataloaders.dataset[view_idx]
            for k in sample.keys():
                try:
                    sample[k] = sample[k].cuda()[None]
                except:
                    pass
            output = self(sample)
            rgb = (output['comp_rgb'][0] * 255).detach().cpu().numpy().astype(np.uint8)
            rgb = Image.fromarray(rgb)
            mask, depth = self.geometry.project_pc(sample['c2w'], H=512, W=512)
            mask = ~mask[0].cpu().numpy()
            mask = Image.fromarray(mask)
            c2w = sample['c2w']
            rgb, mask = self.geometry.add_pc_from_novel_view(rgb, mask, depth, c2w, save_path=os.path.join(self._save_dir[:-4], f'{cnt}.ply'))
            rgb.save(os.path.join(self._save_dir[:-4], f"outpaint_{cnt}.png"))
            mask.save(os.path.join(self._save_dir[:-4], f"mask_{cnt}.png"))
            cnt += 1
            self.outpaint_view[view_idx] = torch.tensor(np.array(rgb), device='cuda')[None] / 255
            self.outpaint_mask[view_idx] = torch.tensor(np.array(mask).astype(float), device='cuda')[None, ..., None]
 
    def distill_language_feature(self) -> None:
        threestudio.info("Start distilling language feature.")
        self.geometry._opacity_mask = None
        total_embed = []
        total_feat = []
        total_flag = []

        for idx in trange(0, len(self.trainer.val_dataloaders.dataset), self.cfg.distill_interval):
            sample = self.trainer.val_dataloaders.dataset[idx]
            for k in sample.keys():
                try:
                    sample[k] = sample[k].cuda()[None]
                except:
                    pass
            output = self(sample)
            rgb = output['comp_rgb']    #shape: 1, 512, 512, 3
            rgb = (rgb.permute(0, 3, 1, 2) * 255).type(torch.uint8) 
            
            try:
                embed, seg, mask= self.sam_clip(rgb)   # feat's shape: N * H * W
                total_embed.append(embed) 
                total_feat.append(seg)
                total_flag.append(idx)
            except:
                threestudio.info(f'except caught during language distillation at {idx}')
                pass 

        # train VAE
        threestudio.info("Start training autoencoder.")
        dataset = Autoencoder_dataset(torch.cat(total_embed, 0).float().numpy())
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=False)
        optimizer = torch.optim.Adam(self.sam_clip_ae.parameters(), lr=self.cfg.sam_clip_ae_lr)
        
        self.sam_clip_ae.train()
        for epoch in tqdm(range(self.cfg.ae_epoch)):
            for idx,  data in enumerate(dataloader):
                data = data.cuda()
                mid = self.sam_clip_ae.encode(data)
                _data = self.sam_clip_ae.decode(mid)
                l2loss = l2_loss(_data, data)
                cosloss = cos_loss(_data, data)
                loss = l2loss + cosloss * 0.001
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.sam_clip_ae.eval()
        mids = dict()
        with torch.no_grad():
            zero_tensor = torch.zeros([1, 512], dtype=float)
            for idx, seg, embed in zip(total_flag, total_feat, total_embed):
                embeds = torch.cat([embed, zero_tensor], 0).float().cuda()
                embeds = self.sam_clip_ae.encode(embeds)
                mid = embeds[seg[:]].squeeze(0).reshape(self.dataset.cfg.height, self.dataset.cfg.width, -1)
                mids[idx] = mid
                rgb = ((mid - mid.min()) / (mid.max() - mid.min())).cpu()
                if self.sam_clip.cfg.vis_pca_feature:
                    self.save_image_grid(f"it{self.global_step}-ae/{idx}.png",
                        [
                            {
                                "type": "rgb",
                                "img": rgb,
                                "kwargs": {"data_format": "HWC"},
                            },
                        ],
                        name="ae",
                        step=self.global_step,
                    )

            if self.sam_clip.cfg.vis_pca_feature:
                self.save_img_sequence(
                    f"it{self.global_step}-ae",
                    f"it{self.global_step}-ae",
                    "(\d+)\.png",
                    save_format="mp4",
                    fps=30,
                    name="ae",
                    step=self.global_step,
                )
        
        threestudio.info("Start training Lang feature.")
        # distill lang feature
        self.geometry.lang_training_setup()
        opt = self.geometry.lang_optimizer
        
        idx_list = list(mids.keys())
        sample_dict = dict()

        for idx, sample in enumerate(self.trainer.val_dataloaders.dataset):
            for k in sample.keys():
                try:
                    sample[k] = sample[k].cuda()[None]
                except:
                    pass
            sample_dict[idx] = sample
            
        for epoch in trange(self.cfg.distill_lang_epoch):
            random.shuffle(idx_list)
            for idx in idx_list:
                sample = sample_dict[idx]
                lang = self(sample)["lang"]
                mid = mids[idx][None]
                loss = l2_loss(mid, lang)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if (epoch + 1) % 30 == 0:
                opt.state = collections.defaultdict(dict)
    
        self.renderer.training=False
        with torch.no_grad():
            lang_min, lang_max = None, None
            for idx, sample in sample_dict.items():
                lang = self(sample)["lang"][0]
                if lang_min is None:
                    lang_min, lang_max = lang.min().item(), lang.max().item()
                self.save_image_grid(f"it{self.global_step}-feat/{idx}.png",
                    [
                        {
                            "type": "rgb",
                            "img": lang,
                            "kwargs": {"data_format": "HWC", "data_range": (lang_min, lang_max)},
                        },
                    ],
                    name=f"feat",
                    step=self.global_step,
                )
        self.renderer.training=True
                 
        self.save_img_sequence(
            f"it{self.global_step}-feat",
            f"it{self.global_step}-feat",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name=f"feat",
            step=self.global_step,
        )

        self.geometry.training_setup()

        threestudio.info("Use Lang feature to crop pts")
        if self.cfg.crop_with_lang:
            p = 2
            if self.geometry._delete_mask is None:
                self.geometry._delete_mask = torch.ones_like(self.geometry.ooi_masks[0])
            for ooi_idx, ooi_mask in enumerate(self.geometry.ooi_masks):
                threestudio.info(self.geometry.ooi_masks[ooi_idx].sum())
                idx = torch.arange(len(ooi_mask), device='cuda')[ooi_mask.bool()]
                lang_feat = self.geometry.get_language_feature[ooi_mask.bool()]
                lang_feat = lang_feat / (lang_feat.norm(2, dim=-1, keepdim=True) + 0.1)
    
                original_ooi_mask = ooi_mask.clone()
                # filter with color by KMeans
                kmeans = KMeans(n_init='auto', n_clusters=10)
                kmeans.fit(lang_feat.detach().cpu())
                labels = kmeans.labels_
                _ = [(labels==i).sum() for i in np.unique(labels)]
                max_label = _.index(max(_))
                dist = ((kmeans.cluster_centers_ - kmeans.cluster_centers_[max_label:max_label+1]) **2).sum(-1)**.5

                for label, num in enumerate(_):
                    if dist[label] > 0.3:
                        ooi_mask[idx[labels == label]] = False
                        self.geometry._delete_mask[idx[labels == label]] = 0.

                p = 1
                # filter with color by Gaussian
                mean, std = lang_feat.mean(0), lang_feat.std(0)
                outlier = torch.logical_or(lang_feat < mean -  p * std, lang_feat > mean + p * std).sum(-1) > 0
                ooi_mask[idx[outlier]] = False
                self.geometry._delete_mask[idx[outlier]] = 0.

                p = 3
                # filter with RGB by Gaussian
                rgb =self.geometry.get_features[original_ooi_mask.bool()][:, 0]
                mean, std = rgb.mean(0), rgb.std(0)
                outlier = torch.logical_or(rgb < mean -  p * std, rgb > mean + p * std).sum(-1) > 0
                ooi_mask[idx[outlier]] = False
                self.geometry._delete_mask[idx[outlier]] = 0.
     
    def load_state_dict(self, state_dict, strict=True):
        i = 0
        while 1:
            if f'geometry.ooi_masks_{i}' not in state_dict.keys(): 
                break
            self.geometry.register_buffer(f'ooi_masks_{i}', state_dict[f'geometry.ooi_masks_{i}'])
            i += 1
        self.geometry.register_buffer('_delete_mask', state_dict['geometry._delete_mask'])
        return super().load_state_dict(state_dict, strict)
