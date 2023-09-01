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

"""
Implementation of NeRO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type
from pathlib import Path

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.ray_samplers import NeuSSampler
from nerfstudio.models.base_surface_model import SurfaceModel, SurfaceModelConfig
from nerfstudio.field_components.spatial_distortions import SceneContraction

from nerfstudio.fields.nero_field import NeROFieldConfig, NeROField
from nerfstudio.utils.custom_eval_utils import eval_setup

@dataclass
class NeROModelConfig(SurfaceModelConfig):
    """NeRO Model Config"""

    _target: Type = field(default_factory=lambda: NeROModel)
    load_config: str = ""
    """Config for model to load"""
    num_samples: int = 64
    """Number of uniform samples"""
    num_samples_importance: int = 64
    """Number of importance samples"""
    num_up_sample_steps: int = 4
    """number of up sample step, 1 for simple coarse-to-fine sampling"""
    base_variance: float = 64
    """fixed base variance in NeuS sampler, the inv_s will be base * 2 ** iter during upsample"""
    perturb: bool = True
    """use to use perturb for the sampled points"""


class NeROModel(SurfaceModel):
    """NeRO model

    Args:
        config: NeRO configuration to instantiate model
    """

    config: NeROModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.sampler = NeuSSampler(
            num_samples=self.config.num_samples,
            num_samples_importance=self.config.num_samples_importance,
            num_samples_outside=self.config.num_samples_outside,
            num_upsample_steps=self.config.num_up_sample_steps,
            base_variance=self.config.base_variance,
        )

        # nero_config = NeROFieldConfig()
        # self.field = NeROField(
        #     config=nero_config,
        #     aabb=self.scene_box.aabb,
        #     spatial_distortion=self.scene_contraction,
        #     num_images=self.num_train_data,
        #     use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        # )
        _, pipeline, _, _ = eval_setup(Path(self.config.load_config), load_model=False)

        self.field = NeROField(
            self.scene_box.aabb,
            hidden_dim=64,
            num_levels=2,
            max_res=2048,
            log2_hashmap_size=19,
            hidden_dim_color=64,
            hidden_dim_transient=64,
            spatial_distortion=SceneContraction(order=float("inf")),
            num_images=self.num_train_data,
            use_pred_normals=False,
            use_average_appearance_embedding=True,
            appearance_embedding_dim=32,
            implementation="torch",
            initial_model=pipeline.model.field,
        )

        self.anneal_end = 50000

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        # anneal for cos in NeRO
        if self.anneal_end > 0:

            def set_anneal(step):
                anneal = min([1.0, step / self.anneal_end])
                self.field.set_cos_anneal_ratio(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )

        return callbacks

    def sample_and_forward_field(self, ray_bundle: RayBundle) -> Dict:
        # ray_samples = self.sampler(ray_bundle, sdf_fn=self.field.get_sdf)
        ray_samples = self.sampler(ray_bundle, sdf_fn=self.field.get_sdf)

        field_outputs = self.field(ray_samples, return_alphas=True)
        weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(
            field_outputs[FieldHeadNames.ALPHA]
        )
        bg_transmittance = transmittance[:, -1, :]

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            "bg_transmittance": bg_transmittance,
        }
        return samples_and_field_outputs

    def get_metrics_dict(self, outputs, batch) -> Dict:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            # training statics
            metrics_dict["s_val"] = self.field.deviation_network.get_variance().item()
            metrics_dict["inv_s"] = 1.0 / self.field.deviation_network.get_variance().item()

        return metrics_dict
