import math
from dataclasses import dataclass

import numpy as np
import threestudio
import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer
from threestudio.utils.typing import *

from ..material.gaussian_material import GaussianDiffuseWithPointLightMaterial
from .gaussian_batch_renderer import GaussianBatchRenderer


class Depth2Normal(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delzdelxkernel = torch.tensor(
            [
                [0.00000, 0.00000, 0.00000],
                [-1.00000, 0.00000, 1.00000],
                [0.00000, 0.00000, 0.00000],
            ]
        )
        self.delzdelykernel = torch.tensor(
            [
                [0.00000, -1.00000, 0.00000],
                [0.00000, 0.00000, 0.00000],
                [0.0000, 1.00000, 0.00000],
            ]
        )

    def forward(self, x):
        B, C, H, W = x.shape
        delzdelxkernel = self.delzdelxkernel.view(1, 1, 3, 3).to(x.device)
        delzdelx = F.conv2d(
            x.reshape(B * C, 1, H, W), delzdelxkernel, padding=1
        ).reshape(B, C, H, W)
        delzdelykernel = self.delzdelykernel.view(1, 1, 3, 3).to(x.device)
        delzdely = F.conv2d(
            x.reshape(B * C, 1, H, W), delzdelykernel, padding=1
        ).reshape(B, C, H, W)
        normal = -torch.cross(delzdelx, delzdely, dim=1)
        return normal


@threestudio.register("diff-gaussian-rasterizer-shading")
class DiffGaussian(Rasterizer, GaussianBatchRenderer):
    @dataclass
    class Config(Rasterizer.Config):
        debug: bool = False
        back_ground_color: Tuple[float, float, float] = (1, 1, 1)

    cfg: Config

    def configure(
        self,
        geometry: BaseGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        if not isinstance(material, GaussianDiffuseWithPointLightMaterial):
            raise NotImplementedError(
                "diff-gaussian-rasterizer-shading only support Gaussian material."
            )
        super().configure(geometry, material, background)
        self.normal_module = Depth2Normal()
        self.background_tensor = torch.tensor(
            self.cfg.back_ground_color, dtype=torch.float32, device="cuda"
        )

    def forward(
        self,
        viewpoint_camera,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        override_color=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """
        # use neural background
        bg_color = bg_color * 0

        pc = self.geometry
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        scales = pc.get_scaling
        rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            shs = pc.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        batch_idx = kwargs["batch_idx"]
        rays_d = kwargs["rays_d"][batch_idx]
        rays_o = kwargs["rays_o"][batch_idx]
        # rays_d_flatten: Float[Tensor, "Nr 3"] = rays_d.unsqueeze(0)

        comp_rgb_bg = self.background(dirs=rays_d.unsqueeze(0))

        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        _, H, W = rendered_image.shape

        xyz_map = rays_o + rendered_depth.permute(1, 2, 0) * rays_d
        normal_map = self.normal_module(xyz_map.permute(2, 0, 1).unsqueeze(0))[0]
        normal_map = F.normalize(normal_map, dim=0)
        if pc.cfg.pred_normal:
            pred_normal_map, _, _, _ = rasterizer(
                means3D=means3D,
                means2D=torch.zeros_like(means2D),
                shs=pc.get_normal.unsqueeze(1),
                colors_precomp=None,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )
        else:
            pred_normal_map = None

        light_positions = kwargs["light_positions"][batch_idx, None, None, :].expand(
            H, W, -1
        )

        if pred_normal_map is not None:
            shading_normal = pred_normal_map.permute(1, 2, 0).detach() * 2 - 1
            shading_normal = F.normalize(shading_normal, dim=2)
        else:
            shading_normal = normal_map.permute(1, 2, 0)
        rgb_fg = self.material(
            positions=xyz_map,
            shading_normal=shading_normal,
            albedo=(rendered_image / (rendered_alpha + 1e-6)).permute(1, 2, 0),
            light_positions=light_positions,
        ).permute(2, 0, 1)
        rendered_image = rgb_fg * rendered_alpha + (
            1 - rendered_alpha
        ) * comp_rgb_bg.reshape(H, W, 3).permute(2, 0, 1)
        normal_map = normal_map * 0.5 * rendered_alpha + 0.5
        mask = rendered_alpha > 0.99
        normal_mask = mask.repeat(3, 1, 1)
        normal_map[~normal_mask] = normal_map[~normal_mask].detach()
        rendered_depth[~mask] = rendered_depth[~mask].detach()

        # Retain gradients of the 2D (screen-space) means for batch dim
        if self.training:
            screenspace_points.retain_grad()

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image.clamp(0, 1),
            "normal": normal_map,
            "pred_normal": pred_normal_map,
            "mask": rendered_alpha,
            "depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
