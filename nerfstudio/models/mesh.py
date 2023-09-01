# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvdiffrast.torch as dr
import open3d as o3d

from nerfstudio.cameras.rays import RayBundle, RaySamples

import matplotlib.pyplot as plt
from matplotlib import figure
from sklearn.neighbors import NearestNeighbors

from nerfstudio.exporter.exporter_utils import generate_point_cloud

from nerfstudio.models.base_model import Model, ModelConfig

from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.custom_eval_utils import eval_setup

from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import nerfstudio.utils.poses as pose_utils
from nerfstudio.viewer.server.viewer_elements import ViewerControl, ViewerButton

from nerfstudio.fields.mesh_field import MeshField, SingleCamera

import open3d as o3d


@dataclass
class MeshModelConfig(ModelConfig):
    """Mesh Model Config"""

    _target: Type = field(default_factory=lambda: MeshModel)
    load_config: str = ""
    """Config for model to load"""
    mesh: str = ""
    """Filepath for mesh to load"""
    use_rgb_model: bool = True
    """Use learnt rgb model"""
    shading_mode: Literal["plain", "lambertian", "nero"] = "plain"
    """Shading mode"""
    rgb_loss_mult: float = 1e0
    """RGB loss multiplier"""
    vertex_loss_mult: float = 0
    """Vertex offset loss multiplier"""
    lap_loss_mult: float = 1e1
    """Laplacian loss multiplier"""
    learn_vertices: bool = True
    """Whether or not to learn vertex offsets"""
    weight_norm: bool = False
    """Whether to use weight normalization in MLPs"""
    geo_feat_dim: int = 256
    hash_encoding: bool = True
    save_dir: str = ""
    """Save directory"""


class MeshModel(Model):
    config: MeshModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        _, pipeline, _, _ = eval_setup(Path(self.config.load_config), load_model=False)
        self.cameras = pipeline.datamanager.cameras

        mesh = o3d.io.read_triangle_mesh(self.config.mesh)
        print(np.array(mesh.triangles).shape)
        # mesh = mesh.subdivide_loop(number_of_iterations=2)
        # mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=200000)
        print(np.array(mesh.triangles).shape)

        self.config.aabb = self.scene_box.aabb
        self.config.num_train_data = self.num_train_data
        self.field = MeshField(mesh, self.config, pipeline.model.field)

        self.viewer_control = ViewerControl()

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        return callbacks

    def get_outputs_for_camera(self, camera: Cameras):
        outputs = self.field.get_camera_outputs(SingleCamera(camera, -1), False, "viewer")
        return outputs

    def get_outputs_for_camera_reshaped(self, camera: Cameras):
        outputs = self.field.get_camera_outputs(SingleCamera(camera, -1), False, "viewer")
        image_height, image_width = outputs["dim"]
        image_outputs = {}
        for key in outputs.keys():
            if key == "dim":
                continue
            image_outputs[key] = outputs[key].view(image_height, image_width, -1)
        return image_outputs

    def get_outputs(self, ray_bundle: RayBundle):
        idx = ray_bundle.camera_indices[0, 0].item()
        # outputs = self.field.get_camera_outputs(SingleCamera(self.cameras[idx], idx), False, str(idx))
        outputs = self.field.get_bundle_outputs(ray_bundle, SingleCamera(self.cameras[idx], idx), False)
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.config.rgb_loss_mult * self.rgb_loss(image, outputs["rgb"])  # 1e0
        if self.training:
            loss_dict["vertex_offset_loss"] = self.config.vertex_loss_mult * self.field.vertex_offset_loss()  # 1e-1
            loss_dict["laplacian_loss"] = self.config.lap_loss_mult * self.field.laplacian_smooth_loss()  # 1e2
            # loss_dict["rgb_variation_loss"] = 0.01 * self.field.rgb_variation_loss()
        print("LOSS", loss_dict)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        pass

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        # param_groups["fields"] = list(self.field.parameters())
        # param_groups["rgb_params"] = list(self.field.rgb_model.parameters())
        param_groups["rgb_params"] = list(self.field.rgb_model.parameters()) + [self.field.rgbs]
        if self.config.learn_vertices:
            param_groups["vertex_offsets"] = [self.field.vertex_offsets]
        return param_groups

    def save_mesh(self, path: Path):
        self.field.save_mesh(path)

    def moo(self) -> bool:
        return True
