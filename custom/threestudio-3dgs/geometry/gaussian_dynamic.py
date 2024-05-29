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
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple

import numpy as np
import threestudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from threestudio.models.geometry.base import BaseGeometry
from threestudio.utils.misc import C
from threestudio.utils.typing import *

from .gaussian_base import GaussianBaseModel


@threestudio.register("gaussian-splatting-dynamic")
class GaussianDynamicModel(GaussianBaseModel):
    @dataclass
    class Config(GaussianBaseModel.Config):
        flow: bool = True
        num_frames: int = 10
        delta_pos_lr: float = 0.001
        delta_rot_lr: float = 0.0001

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self._delta_xyz = torch.empty(0)
        self._delta_rot = torch.empty(0)
        self.time_index = 0

    def training_setup(self):
        super().training_setup()
        l = self.optimize_list
        training_args = self.cfg
        l.append(
            {
                "params": [self._delta_xyz],
                "lr": C(training_args.delta_pos_lr, 0, 0),
                "name": "normal",
            },
        )
        l.append(
            {
                "params": [self._delta_rot],
                "lr": C(training_args.delta_rot_lr, 0, 0),
                "name": "normal",
            },
        )

    @property
    def get_rotation(self):
        return self.rotation_activation(
            self._rotation + self._delta_rot[self.time_index]
        )

    @property
    def get_xyz(self):
        return self._xyz + self._delta_xyz[self.time_index]
