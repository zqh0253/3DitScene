#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math
import os
import random
import sys
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple

import numpy as np
import cv2
from PIL import Image
import threestudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from transformers import pipeline
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
import diffusers
from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
from threestudio.models.geometry.base import BaseGeometry
from threestudio.utils.misc import C
from threestudio.utils.typing import *
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt

from .gaussian_io import GaussianIO
import imageio

from scipy.spatial.transform import Rotation as R

REORDER_MTX = torch.tensor([
    [0,0,0,1],
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0]
]).cuda().float()

def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def rotation_matrix(angle_x, angle_y, angle_z):
    # Convert angles to radians
    rad_x = torch.deg2rad(torch.tensor(angle_x))
    rad_y = torch.deg2rad(torch.tensor(angle_y))
    rad_z = torch.deg2rad(torch.tensor(angle_z))

    # Compute sine and cosine of the angles
    cos_x = torch.cos(rad_x)
    sin_x = torch.sin(rad_x)
    cos_y = torch.cos(rad_y)
    sin_y = torch.sin(rad_y)
    cos_z = torch.cos(rad_z)
    sin_z = torch.sin(rad_z)

    # Construct the rotation matrix
    Rx = torch.tensor([[1, 0, 0],
                   [0, cos_x, -sin_x],
                   [0, sin_x, cos_x]])

    Ry = torch.tensor([[cos_y, 0, sin_y],
                   [0, 1, 0],
                   [-sin_y, 0, cos_y]])

    Rz = torch.tensor([[cos_z, -sin_z, 0],
                   [sin_z, cos_z, 0],
                   [0, 0, 1]])

    # Combine the rotation matrices
    rotation_matrix = Rz @ Ry @ Rx

    return rotation_matrix

# from scipy.spatial import KDTree
# 
# def distCUDA2(points):
#     points_np = points.detach().cpu().float().numpy()
#     dists, inds = KDTree(points_np).query(points_np, k=4)
#     meanDists = (dists[:, 1:] ** 2).mean(1)
# 
#     return torch.tensor(meanDists, dtype=points.dtype, device=points.device)

sys.path.append('./utils/GeoWizard/geowizard')
from models.geowizard_pipeline import DepthNormalEstimationPipeline

C0 = 0.28209479177387814

def propagate(canvas):
    H, W = canvas.shape
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    count = np.zeros_like(canvas)

    while 1:
        curr_mask = canvas > 0
        if sum(sum(curr_mask)) == H * W:
            break
        expand_mask = (cv2.blur(curr_mask.astype(np.float32), (3, 3)) > 0)
        x, y = np.where(np.logical_and(expand_mask, ~curr_mask))
        old_canvas = canvas.copy()

        for xx, yy in zip(x, y):
            for i in range(4):
                ref_x = xx + dx[i]
                ref_y = yy + dy[i]
                if 0<=ref_x<H and 0<=ref_y<W and old_canvas[ref_x, ref_y] != 0:
                    canvas[xx, yy] = old_canvas[ref_x, ref_y]
                    count[xx, yy] = count[ref_x, ref_y] + 1

    weight = (count.max() - count) / count.max()
    return canvas * weight

def save_pc(save_file, pts, color):
    '''
        pts: N, 3
        color: N, 3
    '''
    if color.dtype == np.dtype('float64'):
        color = (color * 255).astype(np.uint8) 
    with open(save_file, 'w') as f:
        f.writelines((
             "ply\n",
             "format ascii 1.0\n",
             "element vertex {}\n".format(pts.shape[0]),
             "property float x\n",
             "property float y\n",
             "property float z\n",
             "property uchar red\n",
             "property uchar green\n",
             "property uchar blue\n",
             "end_header\n"))
        for i in range(pts.shape[0]):
            point = "%f %f %f %d %d %d\n" % (pts[i, 0], pts[i, 1], pts[i, 2], color[i, 0], color[i, 1], color[i, 2]) 
            f.writelines(point)
    threestudio.info(f"Saved point cloud to {save_file}.")


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = (
        covs[:, 0],
        covs[:, 1],
        covs[:, 2],
        covs[:, 3],
        covs[:, 4],
        covs[:, 5],
    )

    # eps must be small enough !!!
    inv_det = 1 / (
        a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24
    )
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = (
        -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f)
        - x * y * inv_b
        - x * z * inv_c
        - y * z * inv_e
    )

    power[power > 0] = -1e10  # abnormal values... make weights 0

    return torch.exp(power)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(
                                str(datetime.now().strftime("%d/%m %H:%M:%S"))
                            ),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


class Camera(NamedTuple):
    FoVx: torch.Tensor
    FoVy: torch.Tensor
    camera_center: torch.Tensor
    image_width: int
    image_height: int
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor

def fill_mask(mask):
    mask = np.array(mask)
    canvas = np.zeros_like(mask)
    H, W = mask.shape
    for i in range(H):
        for p in range(0, W):
            if mask[i, p]:
                canvas[i, p] = 1
            else:
                break
        for p in range(W-1, 0, -1):
            if mask[i, p]:
                canvas[i, p] = 1
            else:
                break
                
    for i in range(W):
        for p in range(0, H):
            if mask[p, i]:
                canvas[p, i] = 1
            else:
                break
        for p in range(H-1, 0, -1):
            if mask[p, i]:
                canvas[p, i] = 1
            else:
                break
    mask = np.logical_and(mask, canvas)
    return Image.fromarray(mask)

def parse_wh(wh):
    try:
        W, H = wh
    except:
        W = H = wh
    return W, H

@threestudio.register("gaussian-splatting")
class GaussianBaseModel(BaseGeometry, GaussianIO):
    @dataclass
    class Config(BaseGeometry.Config):
        max_num: int = 500000
        sh_degree: int = 0
        position_lr: Any = 0.001
        # scale_lr: Any = 0.003
        feature_lr: Any = 0.01
        opacity_lr: Any = 0.05
        scaling_lr: Any = 0.005
        rotation_lr: Any = 0.005
        pred_normal: bool = False
        normal_lr: Any = 0.001
        lang_lr: float = 0.005

        densification_interval: int = 50
        prune_interval: int = 50
        opacity_reset_interval: int = 100000
        densify_from_iter: int = 100
        prune_from_iter: int = 100
        densify_until_iter: int = 2000
        prune_until_iter: int = 2000
        densify_grad_threshold: Any = 0.01
        min_opac_prune: Any = 0.005
        split_thresh: Any = 0.02
        radii2d_thresh: Any = 1000

        sphere: bool = False
        prune_big_points: bool = False
        color_clip: Any = 2.0

        geometry_convert_from: str = ""
        load_ply_only_vertex: bool = False
        init_num_pts: int = 100
        pc_init_radius: float = 0.8
        opacity_init: float = 0.1

        img_resolution: Any = 512

        shap_e_guidance_config: dict = field(default_factory=dict)

        max_scaling: float = 100
        sam_ckpt_path: str = "ckpts/sam_vit_h_4b8939.pth"
        ooi_bbox: Any = None
  
        prompt: Any = None
        empty_prompt: Any = None
        lang_beta_1: float = 0.9
        lang_beta_2: float = 0.999
 
        inference_only: bool = False
        pc_max_resolution: int = 1024
 
        use_sdxl_for_inpaint: bool = False

    cfg: Config

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        self.color_clip = C(self.cfg.color_clip, 0, 0)

        self.fixed_xyz = None
        self.fixed_rot = None

        if not self.cfg.inference_only:
            sam = sam_model_registry["vit_h"](checkpoint=self.cfg.sam_ckpt_path).to('cuda')
            self.predictor = SamPredictor(sam)

    def project_pc(self, c2w, H=256, W=None):
        if W is None:
            W = H
        B = c2w.shape[0]
    
        mask = torch.zeros([B, H, W], device='cuda')
        depth_canvas = torch.zeros([B, H, W], device='cuda')    

        # for pc in [self.bg_point_cloud, self.point_cloud]:
        pc_cam = torch.einsum('bxy,ny->bnx', torch.linalg.inv(c2w), self.point_cloud)
        depth = -1 * pc_cam[..., 2].view(pc_cam.shape[0], -1) 
        pc_cam = (pc_cam / pc_cam[..., 2:3])[..., :3]
        pc_2d = torch.einsum('xy,bny->bnx', self.proj_mtx, pc_cam).clamp(0, 1) 
        pc_2d[..., 0] = pc_2d[..., 0] * (W-1)
        pc_2d[..., 1] = pc_2d[..., 1] * (H-1)
        pc_2d = pc_2d.long()
        for i in range(pc_2d.shape[0]):
            x = (W - pc_2d[i, :, 0]).clamp(0, W-1)
            y = (pc_2d[i, :, 1]).clamp(0, H-1)
            unique_id = x * H + y
            map_2d = np.zeros((W+1)*(H+1)) + 1e8
            np.minimum.at(map_2d, unique_id.cpu(), depth[i].cpu())
            map_2d[map_2d==1e8] = 0
            positive_unique_id = np.where(map_2d>0)[0]
            x, y = positive_unique_id // H, positive_unique_id % H
            mask[i, y, x] = 1.0
            depth_canvas[i, y, x] = torch.tensor(map_2d[positive_unique_id], device='cuda', dtype=torch.float)
                # depth_canvas[i, y, x] = depth[i]

        # pc_cam = torch.einsum('bxy,hwy->bhwx', torch.linalg.inv(c2w), self.point_cloud)
        # depth = -1 * pc_cam[..., 2].view(pc_cam.shape[0], -1) 
        # pc_cam = (pc_cam / pc_cam[..., 2:3])[..., :3]
        # pc_2d = torch.einsum('xy,bhwy->bhwx', self.proj_mtx, pc_cam).clamp(0, 1) 
        # pc_2d[..., 0] = pc_2d[..., 0] * (W-1)
        # pc_2d[..., 1] = pc_2d[..., 1] * (H-1)
        # pc_2d = (pc_2d.long()).view(pc_2d.shape[0], -1, pc_2d.shape[-1])
       
        
        # mask = self.blur_kernel(mask) > 0
        mask = torchvision.transforms.functional.gaussian_blur(mask, 3) > 0
        # mask = mask > 0
        return mask, depth_canvas

    def img2pc_inpaint(self, img, c2w=None, gt_depth=None, mask=None, proj_func=None):
        W, H = parse_wh(self.cfg.img_resolution)
        if max(W, H) > self.cfg.pc_max_resolution:
            W, H = int(W / max(W, H) * self.cfg.pc_max_resolution), int(H / max(W, H) * self.cfg.pc_max_resolution)
        
        with torch.no_grad():
            depth = self.geowizard_pipe(
                img,
                denoising_steps = 25,
                ensemble_size = 3,
                processing_res = 768,
                match_input_res = False,
                domain = 'outdoor',
                color_map = 'Spectral',
                gt_depth = gt_depth, mask = mask,
                show_progress_bar = True)['depth_np']
            ret_depth = depth.copy()
            depth = torch.from_numpy(depth)[None]
            depth = torch.nn.functional.interpolate(depth[None], size=(H, W), mode='bilinear', align_corners=True).squeeze()

        depth = depth.cpu().numpy()
        if proj_func is None:
            depth = depth * 20 + 5
        else:
            depth = proj_func(depth)

        depth = depth * -1 
        x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        x = x / float(W-1)
        y = y / float(H-1)
        xyz = np.stack((x, y, np.ones_like(x)), 0).transpose(1, 2, 0)
        xyz[..., 0] = 1 - xyz[..., 0]

        fov = 60 / 180 * np.pi
        proj_mtx = np.array([
            [1 / (2 * np.tan(fov/2)), 0, 1/2],
            [0, 1 / (2 * np.tan(fov/2)), 1/2],
            [0, 0, 1],
        ])
        self.proj_mtx = torch.from_numpy(proj_mtx).cuda().float()
        if c2w is None:
            c2w = np.array([0.0000, 0.0000, 1.0000, 2.5000, 1.0000, 0.0000, -0.0000, 0.0000, -0.0000, 1.0000, -0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000]).reshape(4, 4)
        else:
            c2w = c2w[0].cpu().numpy()
        xyz = np.einsum('ab,hwb->hwa', np.linalg.inv(proj_mtx), xyz)
        xyz = xyz * depth[..., None]
        xyz = np.concatenate([xyz, np.ones_like(x)[..., None]], 2)
        xyz = np.einsum('ab,hwb->hwa', c2w, xyz)
        return xyz, ret_depth

    def inpaint(self, img, mask, prompt):
        # inpaint using base pipe
        N = 512
        img = img.convert("RGB").resize((N, N))
        mask = mask.convert("RGB").resize((N, N))
        self.base_inpainting_pipe.to("cuda")
        img = self.base_inpainting_pipe(prompt=prompt, image=img, mask_image=mask, guidance_scale=7.5).images[0]
        self.base_inpainting_pipe.to("cpu")
        torch.cuda.empty_cache()

        if self.cfg.use_sdxl_for_inpaint:
            # inpaint using sdxl pipe
            N = 1024
            img = img.convert("RGB").resize((N, N))
            mask = mask.convert("RGB").resize((N, N))
            self.sdxl_inpainting_pipe.to("cuda")
            img = self.sdxl_inpainting_pipe(prompt=prompt, image=img, mask_image=mask, guidance_scale=7.5, num_inference_steps=20, strength=0.99).images[0]
            self.sdxl_inpainting_pipe.to("cpu")
 
        return img

    def configure(self) -> None:
        super().configure()
        self.active_sh_degree = 0
        self.max_sh_degree = self.cfg.sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._opacity_mask = None
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.noise_ratio = 0.0
        if self.cfg.pred_normal:
            self._normal = torch.empty(0)
        self.optimizer = None
        self.setup_functions()
        self.save_path = None
        self.fixed_xyz = None
        self.fixed_rot = None

        if self.cfg.inference_only:
            return 
        # setup GeoWizard
        geowizard_checkpoint_path = 'lemonaddie/geowizard'
        self.geowizard_pipe = DepthNormalEstimationPipeline.from_pretrained(
            geowizard_checkpoint_path, torch_dtype=torch.float32).to(torch.device("cuda"))

        self.base_inpainting_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16)
        # self.base_inpainting_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, safety_checker=None)
        if self.cfg.use_sdxl_for_inpaint:
            self.sdxl_inpainting_pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16")
            self.sdxl_inpainting_pipe.scheduler = diffusers.EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")

        if self.cfg.geometry_convert_from.startswith("depth:"):
            # estimate depth
            W, H = parse_wh(self.cfg.img_resolution)
            if max(W, H) > self.cfg.pc_max_resolution:
                W, H = int(W / max(W, H) * self.cfg.pc_max_resolution), int(H / max(W, H) * self.cfg.pc_max_resolution)
            img = self.cfg.geometry_convert_from[len("depth:"):]
            raw_img = img = Image.open(img).convert("RGB")
            img = img.resize((W, H))

            bg_xyz, bg_color = [], []
            
            with torch.no_grad():
                self.predictor.set_image(np.array(raw_img))
                self.ooi_masks = []
                total_inp_ooi_masks = None
                total_ooi_masks = []
                for i in range(len(self.cfg.ooi_bbox) // 4):
                    bbox = np.array(self.cfg.ooi_bbox[4*i:4*i+4])
                    masks, _, _ = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=bbox[None, :],
                        multimask_output=False,
                    )                
                    # plt.imshow(masks[0])
                    # plt.savefig(os.path.join(self.save_path, f'mask_{i}.png'))
                    ooi_masks = np.array(Image.fromarray(masks[0]).resize((W, H), Image.NEAREST))
                    ooi_masks = (cv2.blur(ooi_masks.astype(np.float32), (5, 5)) > 0)
                    inp_ooi_masks = (cv2.blur(ooi_masks.astype(np.float32), (7, 7)) > 0)
                    if i == 0:
                        total_inp_ooi_masks = inp_ooi_masks
                    else:
                        total_inp_ooi_masks += inp_ooi_masks
                    total_ooi_masks.append(ooi_masks)
      
                total_inp_ooi_masks = total_inp_ooi_masks > 0
                original_wh = parse_wh(self.cfg.img_resolution)
                bg_image = self.inpaint(img=img, mask=Image.fromarray(total_inp_ooi_masks), prompt=self.cfg.empty_prompt).resize((original_wh))
                self.bg_image = np.array(bg_image)
                self.bg_image_mask = np.array(Image.fromarray(total_inp_ooi_masks).resize((original_wh)))
 
            xyz, depth = self.img2pc_inpaint(img)
            self.point_cloud = torch.from_numpy(xyz).cuda().float().reshape(-1, 4)
            
            for ooi_masks in total_ooi_masks:
                transit_masks = np.logical_and(cv2.blur(ooi_masks.astype(np.float32), (3, 3)) > 0, ~ooi_masks)
                depth_tensor = torch.from_numpy(depth)[None, None].cuda() * 2 - 1
                self.ooi_masks.append(torch.tensor(ooi_masks.reshape(-1).astype(np.uint8), device='cuda').float().bool())
                ooi_masks = cv2.blur(ooi_masks.astype(np.float32), (9, 9)) > 0
                mask = torch.from_numpy(ooi_masks.astype(np.float32))[None, None].cuda()
                bg_xyz_pc, _ = self.img2pc_inpaint(bg_image, gt_depth=depth_tensor, mask=1-mask)

                bg_xyz.append(bg_xyz_pc[ooi_masks])
                bg_color.append(np.array(bg_image.resize((W, H)))[ooi_masks] / 255)
    
            # xyz = xyz[..., :3].reshape(-1, 3)
            xyz = xyz.reshape(-1, 4)
            color = np.array(img).reshape(-1, 3) / 255
            bg_xyz = np.concatenate(bg_xyz, 0)
            additional_pts_num = bg_xyz.shape[0]
            xyz = np.concatenate([xyz, bg_xyz], 0)
            self.point_cloud = torch.from_numpy(xyz).cuda().float()

            color = np.concatenate([color, np.concatenate(bg_color, 0)], 0)
            for i in range(len(self.ooi_masks)):
                self.register_buffer(f"ooi_masks_{i}", torch.cat([self.ooi_masks[i], torch.zeros([additional_pts_num], device='cuda').bool()]) )
                self.ooi_masks[i] = getattr(self, f"ooi_masks_{i}")
            self.register_buffer(f"_delete_mask", torch.ones_like(self.ooi_masks[0].float()))

            # project to 3D space
            xyz = xyz[:, :3]
            color = color
            pcd = BasicPointCloud(
                points=xyz, colors=color, normals=np.zeros((xyz.shape[0], 3))
            )
            self.create_from_pcd(pcd, 10)
            self.training_setup()

        elif self.cfg.geometry_convert_from.startswith("shap-e:"):
            shap_e_guidance = threestudio.find("shap-e-guidance")(
                self.cfg.shap_e_guidance_config
            )
            prompt = self.cfg.geometry_convert_from[len("shap-e:") :]
            xyz, color = shap_e_guidance(prompt)

            pcd = BasicPointCloud(
                points=xyz, colors=color, normals=np.zeros((xyz.shape[0], 3))
            )
            self.create_from_pcd(pcd, 10)
            self.training_setup()

        # Support Initialization from OpenLRM, Please see https://github.com/Adamdad/threestudio-lrm
        elif self.cfg.geometry_convert_from.startswith("lrm:"):
            lrm_guidance = threestudio.find("lrm-guidance")(
                self.cfg.shap_e_guidance_config
            )
            prompt = self.cfg.geometry_convert_from[len("lrm:") :]
            xyz, color = lrm_guidance(prompt)

            pcd = BasicPointCloud(
                points=xyz, colors=color, normals=np.zeros((xyz.shape[0], 3))
            )
            self.create_from_pcd(pcd, 10)
            self.training_setup()

        elif os.path.exists(self.cfg.geometry_convert_from):
            threestudio.info(
                "Loading point cloud from %s" % self.cfg.geometry_convert_from
            )
            if self.cfg.geometry_convert_from.endswith(".ckpt"):
                ckpt_dict = torch.load(self.cfg.geometry_convert_from)
                num_pts = ckpt_dict["state_dict"]["geometry._xyz"].shape[0]
                pcd = BasicPointCloud(
                    points=np.zeros((num_pts, 3)),
                    colors=np.zeros((num_pts, 3)),
                    normals=np.zeros((num_pts, 3)),
                )
                self.create_from_pcd(pcd, 10)
                self.training_setup()
                new_ckpt_dict = {}
                for key in self.state_dict():
                    if ckpt_dict["state_dict"].__contains__("geometry." + key):
                        new_ckpt_dict[key] = ckpt_dict["state_dict"]["geometry." + key]
                    else:
                        new_ckpt_dict[key] = self.state_dict()[key]
                self.load_state_dict(new_ckpt_dict)
            elif self.cfg.geometry_convert_from.endswith(".ply"):
                if self.cfg.load_ply_only_vertex:
                    plydata = PlyData.read(self.cfg.geometry_convert_from)
                    vertices = plydata["vertex"]
                    positions = np.vstack(
                        [vertices["x"], vertices["y"], vertices["z"]]
                    ).T
                    if vertices.__contains__("red"):
                        colors = (
                            np.vstack(
                                [vertices["red"], vertices["green"], vertices["blue"]]
                            ).T
                            / 255.0
                        )
                    else:
                        shs = np.random.random((positions.shape[0], 3)) / 255.0
                        C0 = 0.28209479177387814
                        colors = shs * C0 + 0.5
                    normals = np.zeros_like(positions)
                    pcd = BasicPointCloud(
                        points=positions, colors=colors, normals=normals
                    )
                    self.create_from_pcd(pcd, 10)
                else:
                    self.load_ply(self.cfg.geometry_convert_from)
                self.training_setup()
        else:
            threestudio.info("Geometry not found, initilization with random points")
            num_pts = self.cfg.init_num_pts
            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            radius = self.cfg.pc_init_radius * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)

            shs = np.random.random((num_pts, 3)) / 255.0
            C0 = 0.28209479177387814
            color = shs * C0 + 0.5
            pcd = BasicPointCloud(
                points=xyz, colors=color, normals=np.zeros((num_pts, 3))
            )

            self.create_from_pcd(pcd, 10)
            self.training_setup()

    def add_pc_from_novel_view(self, rgb, mask, depth, c2w, save_path=None):
        W, H = parse_wh(self.cfg.img_resolution)
        if max(W, H) > self.cfg.pc_max_resolution:
            W, H = int(W / max(W, H) * self.cfg.pc_max_resolution), int(H / max(W, H) * self.cfg.pc_max_resolution)
        # depth estimation -> add points.
        mask = fill_mask(mask)
        blur_mask = Image.fromarray(cv2.blur(np.array(mask).astype(np.float32), (7, 7)) > 0)
        res = self.inpaint(img=rgb, mask=blur_mask, prompt=self.side_prompt)

        depth_unaligned = self.geowizard_pipe(
                res,
                denoising_steps = 25,
                ensemble_size = 3,
                processing_res = 768,
                match_input_res = False,
                domain = 'outdoor',
                color_map = 'Spectral',
                gt_depth = None, mask = None,
                show_progress_bar = True)['depth_np']
        prev_depth = depth_unaligned[~np.array(mask.resize((768,768)))]
        # inpaint the depth map
        depth_nd = depth[0].cpu().numpy().astype(np.uint8)
        inpaint_mask = np.logical_and(~np.array(mask) , depth[0].cpu().numpy().astype(np.uint8)==0 ).astype(np.uint8)   
        l, r = depth[depth>0].min().item(), depth.max().item()
        depth = (depth - l) / (r - l) * 255
        depth = cv2.inpaint(depth[0].cpu().numpy().astype(np.uint8), inpaint_mask, 3, cv2.INPAINT_TELEA)
        depth = torch.tensor(depth)[None].cuda().float() / 255 
        reproj_func = lambda x: (x - prev_depth.min().item()) / (prev_depth.max().item() - prev_depth.min().item()) * (r-l) + l
        depth = depth * (prev_depth.max() - prev_depth.min()) + prev_depth.min()
        depth_tensor = torch.nn.functional.interpolate(depth[None].cuda(), 768, mode='nearest') * 2 - 1
        
        _masks = cv2.blur(np.array(mask.resize((768, 768))).astype(float), (20, 20)) > 0
        mask_tensor = torch.from_numpy(_masks.astype(np.float32))[None, None].cuda()
        bg_xyz_pc, _ = self.img2pc_inpaint(res, gt_depth=depth_tensor, mask=1-mask_tensor, proj_func=reproj_func, c2w=c2w)

        mask = np.array(Image.fromarray(_masks).resize((W, H)))
        new_xyz = bg_xyz_pc[mask][:, :3]
        res = res.resize((W, H))
        new_color = np.array(res)[mask] / 255
        pcd = BasicPointCloud(points=new_xyz, colors=new_color, normals=np.zeros((new_xyz.shape[0], 3)))
        self.merge_from_pcd(pcd, 10)

        original_wh = parse_wh(self.cfg.img_resolution)
        return res.resize((original_wh)), Image.fromarray(_masks).resize((original_wh))

    @property
    def get_scaling(self):
        if self.cfg.sphere:
            return self.scaling_activation(
                torch.mean(self._scaling, dim=-1).unsqueeze(-1).repeat(1, 3)
            ).clip(0, self.cfg.max_scaling)
        return self.scaling_activation(self._scaling).clip(0, self.cfg.max_scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_language_feature(self):
        return self._language_feature
 
    @property
    def get_xyz(self):
        ret = self._xyz
        if self.noise_ratio > 0.0:
           offset = torch.zeros_like(ret)
           for idx in range(len(self.ooi_masks)):
               ooi_masks = getattr(self, f"ooi_masks_{idx}")
               offset[ooi_masks] = torch.rand(3, device='cuda') * self.noise_ratio
        return ret

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_dc = features_dc.clip(-self.color_clip, self.color_clip)
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        if self._opacity_mask is None:
            ret = self.opacity_activation(self._opacity) 
        else:
            ret = self.opacity_activation(self._opacity) * self._opacity_mask.unsqueeze(-1) 

        if self._delete_mask is None:
            return ret
        else:
            return ret * self._delete_mask.unsqueeze(-1)
       
    @property
    def get_normal(self):
        if self.cfg.pred_normal:
            return self._normal
        else:
            raise ValueError("Normal is not predicted")

    def recover_xyzrot(self):
        self._xyz = torch.nn.Parameter(self.fixed_xyz)
        self._rotation = torch.nn.Parameter(self.fixed_rot)

    def random_rotate(self, rotate_aug_scale, apply_rotate):
        if self.fixed_xyz is None:
            self.fixed_xyz = self.get_xyz.data
            self.fixed_rot = self.get_rotation.data

        if apply_rotate:
            ooi_mask = self.ooi_masks_0.view(-1).byte().to(device='cuda').float()

            rotate = random.randint(-rotate_aug_scale, rotate_aug_scale)
            rot_matrix = rotation_matrix(0, 0, rotate).cuda()
            prev_xyz = self.fixed_xyz.clone()
            ooi_xyz = prev_xyz[ooi_mask.bool()]
            mean = ooi_xyz.mean(0)
            ooi_xyz = ooi_xyz - mean
            after_xyz = torch.einsum('ab,nb->na', rot_matrix, ooi_xyz) + mean
            prev_xyz[ooi_mask.bool()] = after_xyz
            self._xyz = torch.nn.Parameter(prev_xyz)

            prev_rotation = self.fixed_rot.clone()
            prev_rotation_mtx = build_rotation(prev_rotation)
            after_rotation_mtx = torch.einsum('ab,nbc->nac', rot_matrix, prev_rotation_mtx)
            after_rotation = torch.from_numpy(R.from_matrix(after_rotation_mtx.detach().cpu()).as_quat()).cuda().float()
            after_rotation = torch.einsum('ab,nb->na', REORDER_MTX, after_rotation)
            prev_rotation[ooi_mask.bool()] = after_rotation[ooi_mask.bool()]
            self._rotation = torch.nn.Parameter(prev_rotation)
        else:
            self.recover_xyzrot()

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        threestudio.info(
            f"Number of points at initialisation:{fused_point_cloud.shape[0]}"
        )

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            self.cfg.opacity_init
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        if self.cfg.pred_normal:
            normals = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
            self._normal = nn.Parameter(normals.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

        self.fused_point_cloud = fused_point_cloud.cpu().clone().detach()
        self.features = features.cpu().clone().detach()
        self.scales = scales.cpu().clone().detach()
        self.rots = rots.cpu().clone().detach()
        self.opacities = opacities.cpu().clone().detach()

        language_feature = torch.zeros((self._xyz.shape[0], 3), device="cuda")
        self._language_feature = torch.nn.Parameter(language_feature.requires_grad_(True))

    def merge_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        threestudio.info(
            f"Number of points at merging:{fused_point_cloud.shape[0]}"
        )

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            self.cfg.opacity_init
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )
        self.densification_postfix(
            fused_point_cloud,
            features[:, :, 0:1].transpose(1, 2).contiguous(),
            features[:, :, 1:].transpose(1, 2).contiguous(),
            opacities,
            scales,
            rots,
            None,
            torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")    
        )

        for idx in range(len(self.ooi_masks)):
            # self.ooi_masks[idx] = torch.cat([self.ooi_masks[idx], torch.ones([fused_point_cloud.shape[0]], device='cuda') > 0])
            self.register_buffer(f"ooi_masks_{idx}", torch.cat([getattr(self, f"ooi_masks_{idx}"), torch.zeros([fused_point_cloud.shape[0]], device='cuda').bool()]) )
            self.ooi_masks[idx] = getattr(self, f"ooi_masks_{idx}")
        self.register_buffer(f"_delete_mask", torch.ones_like(self.ooi_masks[0].float()))

        # self._xyz = torch.nn.Parameter(torch.cat([self._xyz, fused_point_cloud],0),requires_grad=True)
        # self._features_dc = torch.nn.Parameter(torch.cat([self._features_dc, features[:, :, 0:1].transpose(1, 2).contiguous()],0),requires_grad=True)
        # self._features_rest = torch.nn.Parameter(torch.cat([self._features_rest, features[:, :, 1:].transpose(1, 2).contiguous()],0),requires_grad=True)
        # self._scaling = torch.nn.Parameter(torch.cat([self._scaling, scales],0),requires_grad=True)
        # self._rotation = torch.nn.Parameter(torch.cat([self._rotation, rots],0),requires_grad=True)
        # self._opacity = torch.nn.Parameter(torch.cat([self._opacity, opacities],0),requires_grad=True)

        # if self.cfg.pred_normal:
        #     normals = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
        #     self._normal = nn.Parameter(normals.requires_grad_(True))
        # self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

        # self.fused_point_cloud = fused_point_cloud.cpu().clone().detach()
        # self.features = features.cpu().clone().detach()
        # self.scales = scales.cpu().clone().detach()
        # self.rots = rots.cpu().clone().detach()
        # self.opacities = opacities.cpu().clone().detach()

        # language_feature = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
        # self._language_feature = torch.nn.Parameter(torch.cat([self._language_feature, language_feature], 0), requires_grad=True)
        # self.training_setup()
   

    def lang_training_setup(self):
        training_args = self.cfg
        l = [
                {'params': [self._language_feature], 'lr': C(training_args.lang_lr, 0, 0)}, 
            ]
        self._xyz.requires_grad_(False)
        self._features_dc.requires_grad_(False)
        self._features_rest.requires_grad_(False)
        self._scaling.requires_grad_(False)
        self._rotation.requires_grad_(False)
        self._opacity.requires_grad_(False)
        self._language_feature.requires_grad_(True)
        # self.lang_optimizer = torch.optim.SGD(l, lr=0.0)
        self.lang_optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, betas=(self.cfg.lang_beta_1, self.cfg.lang_beta_2))
        self.optimize_params = ["lang"]
        self.optimize_list = l

    def after_lang(self):
        self._xyz.requires_grad_(True)
        self._features_dc.requires_grad_(True)
        self._features_rest.requires_grad_(True)
        self._scaling.requires_grad_(True)
        self._rotation.requires_grad_(True)
        self._opacity.requires_grad_(True)
        self._language_feature.requires_grad_(False)

    def training_setup(self):
        self._xyz.requires_grad_(True)
        self._features_dc.requires_grad_(True)
        self._features_rest.requires_grad_(True)
        self._scaling.requires_grad_(True)
        self._rotation.requires_grad_(True)
        self._opacity.requires_grad_(True)
        self._language_feature.requires_grad_(False)
        training_args = self.cfg
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": C(training_args.position_lr, 0, 0),
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": C(training_args.feature_lr, 0, 0),
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": C(training_args.feature_lr, 0, 0) / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": C(training_args.opacity_lr, 0, 0),
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": C(training_args.scaling_lr, 0, 0),
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": C(training_args.rotation_lr, 0, 0),
                "name": "rotation",
            },
            {'params': [self._language_feature], 'lr': C(training_args.lang_lr, 0, 0), "name": "language_feature"}, 
        ]
        if self.cfg.pred_normal:
            l.append(
                {
                    "params": [self._normal],
                    "lr": C(training_args.normal_lr, 0, 0),
                    "name": "normal",
                },
            )

        self.optimize_params = [
            "xyz",
            "f_dc",
            "f_rest",
            "opacity",
            "scaling",
            "rotation",
            "language_feature"
        ]
        self.optimize_list = l
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.lang_optimizer = None

    def merge_optimizer(self, net_optimizer):
        l = self.optimize_list
        for param in net_optimizer.param_groups:
            l.append(
                {
                    "params": param["params"],
                    "lr": param["lr"],
                }
            )
        self.optimizer = torch.optim.Adam(l, lr=0.0)
        return self.optimizer

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if not ("name" in param_group):
                continue
            if param_group["name"] == "xyz":
                param_group["lr"] = C(
                    self.cfg.position_lr, 0, iteration, interpolation="exp"
                )
            if param_group["name"] == "scaling":
                param_group["lr"] = C(
                    self.cfg.scaling_lr, 0, iteration, interpolation="exp"
                )
            if param_group["name"] == "f_dc":
                param_group["lr"] = C(
                    self.cfg.feature_lr, 0, iteration, interpolation="exp"
                )
            if param_group["name"] == "f_rest":
                param_group["lr"] = (
                    C(self.cfg.feature_lr, 0, iteration, interpolation="exp") / 20.0
                )
            if param_group["name"] == "opacity":
                param_group["lr"] = C(
                    self.cfg.opacity_lr, 0, iteration, interpolation="exp"
                )
            if param_group["name"] == "rotation":
                param_group["lr"] = C(
                    self.cfg.rotation_lr, 0, iteration, interpolation="exp"
                )
            if param_group["name"] == "normal":
                param_group["lr"] = C(
                    self.cfg.normal_lr, 0, iteration, interpolation="exp"
                )
        if self.lang_optimizer is not None:
            for param_group in self.lang_optimizer.param_groups:
                if not ("name" in param_group):
                    continue
                if param_group["name"] == "language_feature":
                    param_group["lr"] = C(
                        self.cfg.lang_lr, 0, iteration, interpolation="exp"
                    )
        self.color_clip = C(self.cfg.color_clip, 0, iteration)

    def reset_opacity(self):
        # opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        opacities_new = inverse_sigmoid(self.get_opacity * 0.9)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def to(self, device="cpu"):
        self._xyz = self._xyz.to(device)
        self._features_dc = self._features_dc.to(device)
        self._features_rest = self._features_rest.to(device)
        self._opacity = self._opacity.to(device)
        self._scaling = self._scaling.to(device)
        self._rotation = self._rotation.to(device)
        self._normal = self._normal.to(device)
        self._language_feature = self._language_feature.to(device)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if ("name" in group) and group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if ("name" in group) and (group["name"] in self.optimize_params):
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][mask].requires_grad_(True))
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        group["params"][0][mask].requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._language_feature = optimizable_tensors["language_feature"]
        if self.cfg.pred_normal:
            self._normal = optimizable_tensors["normal"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if ("name" in group) and (group["name"] in self.optimize_params):
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (
                            stored_state["exp_avg_sq"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (group["params"][0], extension_tensor), dim=0
                        ).requires_grad_(True)
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (group["params"][0], extension_tensor), dim=0
                        ).requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_normal=None,
        new_language_feature=None
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "language_feature": new_language_feature,
        }
        if self.cfg.pred_normal:
            d.update({"normal": new_normal})

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._language_feature = optimizable_tensors["language_feature"]
        if self.cfg.pred_normal:
            self._normal = optimizable_tensors["normal"]

        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, N=2):
        n_init_points = self._xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.norm(self.get_scaling, dim=1) > self.cfg.split_thresh,
        )

        # divide N to enhance robustness
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1) / N
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_language_feature = self._language_feature[selected_pts_mask].repeat(N,1)
        if self.cfg.pred_normal:
            new_normal = self._normal[selected_pts_mask].repeat(N, 1)
        else:
            new_normal = None

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_normal,
            new_language_feature
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.norm(self.get_scaling, dim=1) <= self.cfg.split_thresh,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_language_feature = self._language_feature[selected_pts_mask]
        if self.cfg.pred_normal:
            new_normal = self._normal[selected_pts_mask]
        else:
            new_normal = None

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_normal,
            new_language_feature
        )

    def densify(self, max_grad):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad)
        self.densify_and_split(grads, max_grad)

    def prune(self, min_opacity, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if self.cfg.prune_big_points:
            big_points_vs = self.max_radii2D > (torch.mean(self.max_radii2D) * 3)
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    @torch.no_grad()
    def update_states(
        self,
        iteration,
        visibility_filter,
        radii,
        viewspace_point_tensor,
    ):
        if self._xyz.shape[0] >= self.cfg.max_num + 100:
            prune_mask = torch.randperm(self._xyz.shape[0]).to(self._xyz.device)
            prune_mask = prune_mask > self.cfg.max_num
            self.prune_points(prune_mask)
            return
        # Keep track of max radii in image-space for pruning
        # loop over batch
        bs = len(viewspace_point_tensor)
        for i in range(bs):
            radii_i = radii[i]
            visibility_filter_i = visibility_filter[i]
            viewspace_point_tensor_i = viewspace_point_tensor[i]
            self.max_radii2D = torch.max(self.max_radii2D, radii_i.float())

            self.add_densification_stats(viewspace_point_tensor_i, visibility_filter_i)

        if (
            iteration > self.cfg.prune_from_iter
            and iteration < self.cfg.prune_until_iter
            and iteration % self.cfg.prune_interval == 0
        ):
            self.prune(self.cfg.min_opac_prune, self.cfg.radii2d_thresh)
            if iteration % self.cfg.opacity_reset_interval == 0:
                self.reset_opacity()

        if (
            iteration > self.cfg.densify_from_iter
            and iteration < self.cfg.densify_until_iter
            and iteration % self.cfg.densification_interval == 0
        ):
            self.densify(self.cfg.densify_grad_threshold)
