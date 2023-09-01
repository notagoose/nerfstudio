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
Field for SDF based model, rather then estimating density to generate a surface,
a signed distance function (SDF) for surface representation is used to help with extracting high fidelity surfaces
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Type, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, FieldConfig, shift_directions_for_tcnn
from nerfstudio.field_components.mlp import MLP

from nerfstudio.field_components.activations import trunc_exp

from nerfstudio.fields.ref_utils import generate_ide_fn
from nerfstudio.fields.nero_utils import IdentityActivation, ExpActivation, linear_to_srgb
from nerfstudio.fields.sdf_field import LearnedVariance
from nerfstudio.utils.timer import Timer

import nvdiffrast.torch as dr

try:
    import tinycudann as tcnn
except ModuleNotFoundError:
    # tinycudann module doesn't exist
    pass


def find_nan(t):
    return
    # assert not torch.isnan(t).any()


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dims,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj):
        return eo.embed(x)

    return embed, embedder_obj.out_dim

    def forward(self, x):
        return x


class ExpActivation(nn.Module):
    def __init__(self, max_light=5.0):
        super().__init__()
        self.max_light = max_light

    def forward(self, x):
        return torch.exp(torch.clamp(x, max=self.max_light))


class BiasLayer(nn.Module):
    def __init__(self, bias=0):
        super().__init__()
        self.bias = bias

    def forward(self, x):
        return x + self.bias


class SDFNetwork(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        skip_in=(4,),
        multires=0,
        bias=0.5,
        scale=1,
        geometric_init=False,
        weight_norm=True,
        inside_outside=False,
        sdf_activation="none",
        layer_activation="softplus",
    ):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if layer_activation == "softplus":
            self.activation = nn.Softplus(beta=100)
        elif layer_activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError

    def forward(self, inputs):
        print(inputs.min().item(), inputs.max().item(), inputs.shape, "SELF INPUTS")
        inputs = (2 * inputs - 1) * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        print(x.min().item(), x.max().item(), x.shape, "SELF OUTPUTS")
        return x

    def sdf(self, x):
        return self.forward(x)[..., :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        return gradients

    def sdf_normal(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        return y[..., :1].detach(), gradients.detach()


@dataclass
class NeROFieldConfig(FieldConfig):
    """SDF Field Config"""

    _target: Type = field(default_factory=lambda: NeROField)
    num_layers: int = 8
    """Number of layers for geometric network"""
    hidden_dim: int = 64
    """Number of hidden dimension of geometric network"""
    geo_feat_dim: int = 256
    """Dimension of geometric feature"""
    num_layers_color: int = 4
    """Number of layers for color network"""
    hidden_dim_color: int = 64
    """Number of hidden dimension of color network"""
    appearance_embedding_dim: int = 32
    """Dimension of appearance embedding"""
    use_appearance_embedding: bool = False
    """Whether to use appearance embedding"""
    bias: float = 0.8
    """Sphere size of geometric initialization"""
    geometric_init: bool = False
    """Whether to use geometric initialization"""
    inside_outside: bool = False
    """Whether to revert signed distance value, set to True for indoor scene"""
    weight_norm: bool = True
    """Whether to use weight norm for linear layer"""
    use_grid_feature: bool = False
    """Whether to use multi-resolution feature grids"""
    divide_factor: float = 2.0
    """Normalization factor for multi-resolution grids"""
    beta_init: float = 0.1
    """Init learnable beta value for transformation of sdf to density"""
    encoding_type: Literal["hash", "periodic", "tensorf_vm"] = "hash"
    num_levels = 3  # 16
    """Number of encoding levels"""
    max_res = 2048  # 2048
    """Maximum resolution of the encoding"""
    base_res = 5  # 16
    """Base resolution of the encoding"""
    log2_hashmap_size = 8  # 19
    """Size of the hash map"""
    features_per_level = 1
    """Number of features per encoding level"""
    use_hash = False
    """Whether to use hash encoding"""
    smoothstep = True
    """Whether to use the smoothstep function"""


class NeROField(Field):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        hidden_dim_color: dimension of hidden layers for color network
        hidden_dim_transient: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_transient_embedding: whether to use transient embedding
        use_semantics: whether to use semantic segmentation
        num_semantic_classes: number of semantic classes
        use_pred_normals: whether to use predicted normals
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_weight_norm: bool = False,
        hash_encoding: bool = True,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        initial_model: Optional[Field] = None,
        timer: Optional[Timer] = None,
    ) -> None:
        super().__init__()
        self.timer = timer
        if timer is None:
            self.timer = Timer()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.use_weight_norm = use_weight_norm
        if self.use_weight_norm:
            assert implementation == "torch"
        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_transient_embedding = use_transient_embedding
        self.use_semantics = use_semantics
        self.use_pred_normals = use_pred_normals
        self.pass_semantic_gradients = pass_semantic_gradients
        self.base_res = base_res

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=2, min_freq_exp=0, max_freq_exp=2 - 1, implementation=implementation
        )

        self.mlp_base_grid = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )
        self.mlp_base_mlp = MLP(
            in_dim=self.mlp_base_grid.get_out_dim(),
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            weight_norm=self.use_weight_norm,
            implementation=implementation,
        )
        if hash_encoding:
            self.mlp_base = torch.nn.Sequential(self.mlp_base_grid, self.mlp_base_mlp)
        else:
            # self.mlp_base = MLP(in_dim=3,
            #     num_layers=num_layers,
            #     layer_width=hidden_dim,
            #     out_dim=1 + self.geo_feat_dim,
            #     activation=nn.ReLU(),
            #     out_activation=None,
            #     weight_norm=self.use_weight_norm,
            #     implementation=implementation,
            # )
            self.mlp_base = SDFNetwork(
                d_out=1 + self.geo_feat_dim,
                d_in=3,
                d_hidden=128,
                n_layers=8,
                skip_in=[4],
                multires=6,
                bias=0.5,
                scale=1.0,
                geometric_init=False,
                weight_norm=True,
                sdf_activation="none",
            )
        # transients
        if self.use_transient_embedding:
            self.transient_embedding_dim = transient_embedding_dim
            self.embedding_transient = Embedding(self.num_images, self.transient_embedding_dim)
            self.mlp_transient = MLP(
                in_dim=self.geo_feat_dim + self.transient_embedding_dim,
                num_layers=num_layers_transient,
                layer_width=hidden_dim_transient,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_transient_uncertainty = UncertaintyFieldHead(in_dim=self.mlp_transient.get_out_dim())
            self.field_head_transient_rgb = TransientRGBFieldHead(in_dim=self.mlp_transient.get_out_dim())
            self.field_head_transient_density = TransientDensityFieldHead(in_dim=self.mlp_transient.get_out_dim())

        # semantics
        if self.use_semantics:
            self.mlp_semantics = MLP(
                in_dim=self.geo_feat_dim,
                num_layers=2,
                layer_width=64,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_semantics = SemanticFieldHead(
                in_dim=self.mlp_semantics.get_out_dim(), num_classes=num_semantic_classes
            )

        # predicted normals
        if self.use_pred_normals:
            self.mlp_pred_normals = MLP(
                in_dim=self.geo_feat_dim + self.position_encoding.get_out_dim(),
                num_layers=3,
                layer_width=64,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_pred_normals = PredNormalsFieldHead(in_dim=self.mlp_pred_normals.get_out_dim())

        self.mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            weight_norm=self.use_weight_norm,
            implementation=implementation,
        )

        self.metallic_predictor = MLP(
            in_dim=self.geo_feat_dim + 3,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=1,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            weight_norm=self.use_weight_norm,
            implementation=implementation,
        )

        self.albedo_predictor = MLP(
            in_dim=self.geo_feat_dim + 3,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            weight_norm=self.use_weight_norm,
            implementation=implementation,
        )

        self.roughness_predictor = MLP(
            in_dim=self.geo_feat_dim + 3,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=1,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            weight_norm=self.use_weight_norm,
            implementation=implementation,
        )

        FG_LUT = torch.from_numpy(
            np.fromfile("../NeRO/assets/bsdf_256_256.bin", dtype=np.float32).reshape(1, 256, 256, 2)
        )
        self.register_buffer("FG_LUT", FG_LUT)

        self.sph_enc = generate_ide_fn(5)
        self.dir_enc, dir_dim = get_embedder(6, 3)
        self.pos_enc, pos_dim = get_embedder(8, 3)
        exp_max = 5

        self.weird_dim = 72
        self.outer_light = nn.Sequential(
            MLP(
                in_dim=self.weird_dim,
                num_layers=num_layers_color,
                layer_width=hidden_dim_color,
                out_dim=3,
                activation=nn.ReLU(),
                out_activation=None,
                weight_norm=self.use_weight_norm,
                implementation=implementation,
            ),
            BiasLayer(np.log(0.5)),
            ExpActivation(max_light=exp_max),
        )

        self.inner_light = nn.Sequential(
            MLP(
                in_dim=self.weird_dim + pos_dim,
                num_layers=num_layers_color,
                layer_width=hidden_dim_color,
                out_dim=3,
                activation=nn.ReLU(),
                out_activation=None,
                weight_norm=self.use_weight_norm,
                implementation=implementation,
            ),
            BiasLayer(np.log(0.5)),
            ExpActivation(max_light=exp_max),
        )

        self.inner_weight = nn.Sequential(
            MLP(
                in_dim=dir_dim + pos_dim,
                num_layers=num_layers_color,
                layer_width=hidden_dim_color,
                out_dim=1,
                activation=nn.ReLU(),
                out_activation=None,
                weight_norm=self.use_weight_norm,
                implementation=implementation,
            ),
            BiasLayer(-0.95),
        )
        # outer lights are direct lights
        # dim_num = 72
        # self.outer_light = make_predictor(dim_num, 3, activation='exp', exp_max=exp_max)
        # nn.init.constant_(self.outer_light[-2].bias,np.log(0.5))

        # inner lights are indirect lights
        # self.inner_light = make_predictor(pos_dim + dim_num, 3, activation='exp', exp_max=exp_max)
        # nn.init.constant_(self.inner_light[-2].bias, np.log(0.5))
        # self.inner_weight = make_predictor(pos_dim + dir_dim, 1, activation='none')
        # nn.init.constant_(self.inner_weight[-2].bias, self.cfg['inner_init'])

        self.sphere_direction = False

        self.deviation_network = LearnedVariance(init_val=0.1)

        self._cos_anneal_ratio = 1.0

        if initial_model is not None:
            self.init_from(initial_model)

    def init_from(self, initial_model):
        self.mlp_base_grid = initial_model.mlp_base_grid
        self.mlp_base_mlp = initial_model.mlp_base_mlp
        self.mlp_base = torch.nn.Sequential(self.mlp_base_grid, self.mlp_base_mlp)
        if self.use_transient_embedding:
            self.transient_embedding_dim = transient_embedding_dim
            self.embedding_transient = Embedding(self.num_images, self.transient_embedding_dim)
            self.mlp_transient = initial_model.mlp_transient

        # semantics
        if self.use_semantics:
            self.mlp_semantics = initial_model.mlp_semantics

        # predicted normals
        if self.use_pred_normals:
            self.mlp_pred_normals = initial_model.mlp_pred_normals

        self.mlp_head = initial_model.mlp_head

    def get_sdf(self, ray_samples: RaySamples) -> Float[Tensor, "num_samples ... 1"]:
        """predict the sdf value for ray samples"""
        positions = ray_samples.frustums.get_start_positions()
        positions_flat = positions.view(-1, 3)
        hidden_output = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        sdf, _ = torch.split(hidden_output, [1, self.geo_feat_dim], dim=-1)
        return sdf

    def set_cos_anneal_ratio(self, anneal: float) -> None:
        """Set the anneal value for the variance."""
        self._cos_anneal_ratio = anneal

    def get_alpha(
        self,
        ray_samples: RaySamples,
        sdf: Optional[Float[Tensor, "num_samples ... 1"]] = None,
        gradients: Optional[Float[Tensor, "num_samples ... 1"]] = None,
    ) -> Float[Tensor, "num_samples ... 1"]:
        """compute alpha from sdf as in NeuS"""
        if sdf is None or gradients is None:
            inputs = ray_samples.frustums.get_start_positions()
            inputs.requires_grad_(True)
            with torch.enable_grad():
                hidden_output = self.mlp_base(inputs)
                sdf, _ = torch.split(hidden_output, [1, self.geo_feat_dim], dim=-1)
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=inputs,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

        inv_s = self.deviation_network.get_variance()  # Single parameter

        true_cos = (ray_samples.frustums.directions * gradients).sum(-1, keepdim=True)

        # anneal as NeuS
        cos_anneal_ratio = self._cos_anneal_ratio

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + F.relu(-true_cos) * cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * ray_samples.deltas * 0.5
        estimated_prev_sdf = sdf - iter_cos * ray_samples.deltas * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        return alpha

    def get_density_at_points(self, positions: Tensor) -> Tuple[Tensor, Tensor]:
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(positions, self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        # temp = self.mlp_base_grid(positions_flat)
        # print("TEMP", temp.min().item(), temp.max().item(), temp.shape, "TEMP")
        h = self.mlp_base(positions_flat).view(*positions.shape[:-1], -1)
        # print("H", h.min().item(), h.max().item(), h.shape, "HM")
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_positions()
        return self.get_density_at_points(positions)

    def get_sdf(self, ray_samples: RaySamples) -> Float[Tensor, "num_samples ... 1"]:
        """predict the sdf value for ray samples"""
        positions = ray_samples.frustums.get_start_positions()
        positions_flat = positions.view(-1, 3)
        hidden_output = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        sdf, _ = torch.split(hidden_output, [1, self.geo_feat_dim], dim=-1)
        return sdf

    def predict_specular_lights(self, points, feature_vectors, reflective, roughness):
        self.timer.start("SPE-A1")
        ref_roughness = self.sph_enc(reflective, roughness, self.timer)
        self.timer.stop()
        self.timer.start("SPE-A2")
        pts = self.pos_enc(2 * points - 1)
        self.timer.stop()
        # if self.sphere_direction:
        #     sph_points = offset_points_to_sphere(points)
        #     sph_points = safe_normalize(sph_points + reflective * get_sphere_intersection(sph_points, reflective))
        #     sph_points = self.sph_enc(sph_points, roughness)
        #     direct_light = self.outer_light(torch.cat([ref_roughness, sph_points], -1))
        # else:
        self.timer.start("SPE-B1")
        direct_light = self.outer_light(ref_roughness)
        light = direct_light
        occ_prob = 0 * light
        indirect_light = 0 * light
        self.timer.stop()

        # indirect_light = self.inner_light(torch.cat([pts, ref_roughness], -1))
        # ref_ = self.dir_enc(reflective)
        # occ_prob = self.inner_weight(torch.cat([pts.detach(), ref_.detach()], -1)) # this is occlusion prob
        # occ_prob = occ_prob*0.5 + 0.5
        # occ_prob_ = torch.clamp(occ_prob,min=0,max=1)

        # print(occ_prob_.mean().item(), "OCC_PROB")

        # light = indirect_light * occ_prob_ + direct_light * (1 - occ_prob_)
        # indirect_light = indirect_light * occ_prob_
        return light, occ_prob, indirect_light

    def predict_diffuse_lights(self, points, feature_vectors, normals):
        self.timer.start("DIF-A1")
        roughness = torch.ones([normals.shape[0], 1], device=normals.device)
        ref = self.sph_enc(normals, roughness, self.timer)
        self.timer.stop()

        # if self.sphere_direction:
        #     sph_points = offset_points_to_sphere(points)
        #     sph_points = safe_normalize(sph_points + normals * get_sphere_intersection(sph_points, normals))
        #     sph_points = self.sph_enc(sph_points, roughness)
        #     light = self.outer_light(torch.cat([ref, sph_points], -1))
        # else:
        self.timer.start("DIF-A2")
        light = self.outer_light(ref)
        self.timer.stop()
        return light

    def get_colors(self, points, normals, view_dirs, feature_vectors):
        # print("SCALE POINTS", points.min().item(), points.max().item(), points.shape)
        # print("SCALE FEATURES", feature_vectors.min().item(), feature_vectors.max().item(), feature_vectors.shape)
        self.timer.start("CLR-A1")
        normals = safe_normalize(normals)
        view_dirs = safe_normalize(view_dirs)
        self.timer.stop()
        self.timer.start("CLR-A2")

        find_nan(normals)
        find_nan(view_dirs)
        self.timer.stop()
        self.timer.start("CLR-A3")
        reflective = torch.sum(view_dirs * normals, -1, keepdim=True) * normals * 2 - view_dirs
        NoV = torch.sum(normals * view_dirs, -1, keepdim=True)
        self.timer.stop()
        self.timer.start("CLR-A4")
        find_nan(reflective)
        find_nan(NoV)

        find_nan(feature_vectors)
        find_nan(points)
        self.timer.stop()

        # print("feature_vectors", feature_vectors.min().item(), feature_vectors.max().item(), feature_vectors.dtype)
        # print("points", points.min().item(), points.max().item(), points.dtype)
        self.timer.start("CLR-B1")
        metallic = self.metallic_predictor(torch.cat([feature_vectors, points], -1))
        self.timer.stop()
        self.timer.start("CLR-B2")
        roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1))
        self.timer.stop()
        self.timer.start("CLR-B3")
        albedo = self.albedo_predictor(torch.cat([feature_vectors, points], -1))
        self.timer.stop()

        # print("metallic", metallic.min().item(), metallic.max().item(), metallic.dtype)
        # print("albedo", albedo.min().item(), albedo.max().item(), albedo.dtype)
        self.timer.start("CLR-B4")
        find_nan(metallic)
        find_nan(roughness)
        find_nan(albedo)
        self.timer.stop()

        # diffuse light
        self.timer.start("CLR-B5")
        diffuse_light = self.predict_diffuse_lights(points, feature_vectors, normals)
        self.timer.stop()

        self.timer.start("CLR-B6")
        diffuse_albedo = (1 - metallic) * albedo
        diffuse_color = diffuse_albedo * diffuse_light
        find_nan(diffuse_albedo)
        find_nan(diffuse_light)
        find_nan(diffuse_color)
        self.timer.stop()

        # return diffuse_albedo

        # specular light
        self.timer.start("CLR-C1")
        specular_albedo = 0.04 * (1 - metallic) + metallic * albedo
        self.timer.stop()
        self.timer.start("CLR-C2")
        specular_light, occ_prob, indirect_light = self.predict_specular_lights(
            points, feature_vectors, reflective, roughness
        )
        self.timer.stop()

        self.timer.start("CLR-D1")
        fg_uv = torch.cat([torch.clamp(NoV, min=0.0, max=1.0), torch.clamp(roughness, min=0.0, max=1.0)], -1)
        pn, bn = points.shape[0], 1
        self.timer.stop()
        self.timer.start("CLR-D2")
        fg_lookup = dr.texture(
            self.FG_LUT, fg_uv.reshape(1, pn // bn, bn, -1).contiguous(), filter_mode="linear", boundary_mode="clamp"
        ).reshape(pn, 2)
        self.timer.stop()
        self.timer.start("CLR-D3")

        specular_ref = specular_albedo * fg_lookup[:, 0:1] + fg_lookup[:, 1:2]
        specular_color = specular_ref * specular_light
        self.timer.stop()

        # integrated together
        self.timer.start("CLR-E1")
        color = diffuse_color + specular_color
        self.timer.stop()

        # gamma correction
        self.timer.start("CLR-E2")
        do_gamma_correction = False
        if do_gamma_correction:
            diffuse_color = linear_to_srgb(diffuse_color)
            specular_color = linear_to_srgb(specular_color)
            color = linear_to_srgb(color)
            color = torch.clamp(color, min=0.0, max=1.0)
        self.timer.stop()

        self.timer.start("CLR-F1")
        occ_info = {
            "reflective": reflective,
            "metallic": metallic,
            "occ_prob": occ_prob,
            "diffuse_color": diffuse_color,
            "diffuse_light": diffuse_light,
            "specular_ref": specular_ref,
            "specular_color": specular_color,
            "specular_light": specular_light,
            "indirect_light": indirect_light,
            "diffuse_albedo": diffuse_albedo,
            "specular_albedo": specular_albedo,
        }
        self.timer.stop()
        return color, occ_info

    def predict_materials(self, points, feature_vectors):
        metallic = self.metallic_predictor(torch.cat([feature_vectors, points], -1))
        roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1))
        albedo = self.albedo_predictor(torch.cat([feature_vectors, points], -1))
        return metallic, roughness, albedo

    def get_outputs_old(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = shift_directions_for_tcnn(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                )

        # transients
        if self.use_transient_embedding and self.training:
            embedded_transient = self.embedding_transient(camera_indices)
            transient_input = torch.cat(
                [
                    density_embedding.view(-1, self.geo_feat_dim),
                    embedded_transient.view(-1, self.transient_embedding_dim),
                ],
                dim=-1,
            )
            x = self.mlp_transient(transient_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.UNCERTAINTY] = self.field_head_transient_uncertainty(x)
            outputs[FieldHeadNames.TRANSIENT_RGB] = self.field_head_transient_rgb(x)
            outputs[FieldHeadNames.TRANSIENT_DENSITY] = self.field_head_transient_density(x)

        # semantics
        if self.use_semantics:
            semantics_input = density_embedding.view(-1, self.geo_feat_dim)
            if not self.pass_semantic_gradients:
                semantics_input = semantics_input.detach()

            x = self.mlp_semantics(semantics_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantics(x)

        # predicted normals
        if self.use_pred_normals:
            positions = ray_samples.frustums.get_positions()

            positions_flat = self.position_encoding(positions.view(-1, 3))
            pred_normals_inp = torch.cat([positions_flat, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

            x = self.mlp_pred_normals(pred_normals_inp).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.PRED_NORMALS] = self.field_head_pred_normals(x)

        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim),
                embedded_appearance.view(-1, self.appearance_embedding_dim),
            ],
            dim=-1,
        )
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs

    def get_outputs(
        self,
        ray_samples: RaySamples,
        density_embedding: Optional[Tensor] = None,
        return_alphas: bool = False,
    ) -> Dict[FieldHeadNames, Tensor]:
        """compute output of ray samples"""
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        outputs = {}

        camera_indices = ray_samples.camera_indices.squeeze()

        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)

        directions = ray_samples.frustums.directions
        directions_flat = directions.reshape(-1, 3)
        d = self.direction_encoding(directions_flat)

        # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                )

        # inputs.requires_grad_(True)
        # with torch.enable_grad():
        #     hidden_output = self.mlp_base(inputs)
        #     sdf, geo_feature = torch.split(hidden_output, [1, self.geo_feat_dim], dim=-1)
        # d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        # gradients = torch.autograd.grad(
        #     outputs=sdf, inputs=inputs, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
        # )[0]
        hidden_output = self.mlp_base(inputs)
        sdf, geo_feature = torch.split(hidden_output, [1, self.geo_feat_dim], dim=-1)

        eps = 1e-5
        gradients_xyz = []
        for i in range(3):
            eps_tensor = torch.zeros_like(inputs)
            eps_tensor[..., i] = eps
            gradients_temp = (
                self.mlp_base(inputs + eps_tensor)[..., :1] - self.mlp_base(inputs - eps_tensor)[..., :1]
            ) / (2 * eps)
            gradients_xyz.append(gradients_temp)
        gradients = torch.cat(gradients_xyz, dim=-1)

        # h = torch.cat(
        #     [
        #         d,
        #         geo_feature.view(-1, self.geo_feat_dim),
        #         embedded_appearance.view(-1, self.appearance_embedding_dim),
        #     ],
        #     dim=-1,
        # )

        rgb, _ = self.get_colors(
            inputs, directions_flat, gradients, geo_feature.view(-1, self.geo_feat_dim)
        )  # , camera_indices)

        rgb = rgb.view(*ray_samples.frustums.directions.shape[:-1], -1)
        sdf = sdf.view(*ray_samples.frustums.directions.shape[:-1], -1)
        gradients = gradients.view(*ray_samples.frustums.directions.shape[:-1], -1)
        normals = torch.nn.functional.normalize(gradients, p=2, dim=-1)

        outputs.update(
            {
                FieldHeadNames.RGB: rgb,
                FieldHeadNames.SDF: sdf,
                FieldHeadNames.NORMALS: normals,
                FieldHeadNames.GRADIENT: gradients,
            }
        )

        if return_alphas:
            alphas = self.get_alpha(ray_samples, sdf, gradients)
            outputs.update({FieldHeadNames.ALPHA: alphas})

        return outputs

    def get_special_outputs(
        self, ray_samples: RaySamples, normal_samples: Tensor, return_alphas: bool = False
    ) -> Dict[FieldHeadNames, Tensor]:
        self.timer.lap()
        self.timer.start("NF-A1")
        density, density_embedding = self.get_density(ray_samples)
        self.timer.stop()
        # print(density.min().item(), density.max().item(), density.shape, "density")
        # print(density_embedding.min().item(), density_embedding.max().item(), density_embedding.shape, "not okay")
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        self.timer.start("NF-B1")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = shift_directions_for_tcnn(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        self.timer.stop()
        self.timer.start("NF-B2")
        d = self.direction_encoding(directions_flat)
        self.timer.stop()

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                )

        # h = torch.cat(
        #     [
        #         d,
        #         density_embedding.view(-1, self.geo_feat_dim),
        #         embedded_appearance.view(-1, self.appearance_embedding_dim),
        #     ],
        #     dim=-1,
        # )
        self.timer.start("NF-C1")
        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)

        directions = ray_samples.frustums.directions
        directions_flat = directions.reshape(-1, 3)

        normal_samples = normal_samples.reshape(-1, 3)
        self.timer.stop()
        self.timer.start("NF-D1")
        # rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        rgb, other = self.get_colors(
            inputs, -directions_flat, normal_samples, density_embedding.view(-1, self.geo_feat_dim)
        )
        self.timer.stop()
        self.timer.start("NF-E1")
        rgb = rgb.view(*ray_samples.frustums.directions.shape[:-1], -1)
        for key in other.keys():
            other[key] = other[key].view(*ray_samples.frustums.directions.shape[:-1], -1)
        outputs.update({FieldHeadNames.RGB: rgb})
        outputs.update(other)
        self.timer.stop()
        return outputs

    def forward(
        self, ray_samples: RaySamples, compute_normals: bool = False, return_alphas: bool = False
    ) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
            compute normals: not currently used in this implementation.
            return_alphas: Whether to return alpha values
        """
        # with torch.cuda.amp.autocast(enabled=False):
        field_outputs = self.get_outputs(ray_samples, return_alphas=return_alphas)
        return field_outputs
