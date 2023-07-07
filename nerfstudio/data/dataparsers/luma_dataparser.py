# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""Data parser for luma dataset"""
from __future__ import annotations
from typing import Literal, Optional, Type

import os

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rich.console import Console

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json

import open3d as o3d

CONSOLE = Console()


def get_src_from_pairs(
    ref_idx, all_imgs, pairs_srcs, neighbors_num=None, neighbors_shuffle=False
):
    # src_idx[0] is ref img
    src_idx = pairs_srcs[ref_idx]
    # randomly sample neighbors
    if neighbors_num and neighbors_num > -1 and neighbors_num < len(src_idx) - 1:
        if neighbors_shuffle:
            perm_idx = torch.randperm(len(src_idx) - 1) + 1
            src_idx = torch.cat([src_idx[[0]], src_idx[perm_idx[:neighbors_num]]])
        else:
            src_idx = src_idx[: neighbors_num + 1]
    src_idx = src_idx.to(all_imgs.device)
    return {"src_imgs": all_imgs[src_idx], "src_idxs": src_idx}


def get_image(image_filename, alpha_color=None):
    """Returns a 3 channel image.

    Args:
        image_idx: The image index in the dataset.
    """
    pil_image = Image.open(image_filename)
    np_image = np.array(pil_image, dtype="uint8")  # shape is (h, w, 3 or 4)
    assert len(np_image.shape) == 3
    assert np_image.dtype == np.uint8
    assert np_image.shape[2] in [3, 4], f"Image shape of {np_image.shape} is in correct."
    image = torch.from_numpy(np_image.astype("float32") / 255.0)
    if alpha_color is not None and image.shape[-1] == 4:
        assert image.shape[-1] == 4
        image = image[:, :, :3] * image[:, :, -1:] + alpha_color * (1.0 - image[:, :, -1:])
    else:
        image = image[:, :, :3]
    return image


def get_depths_and_normals(image_idx: int, depths, normals):
    """function to process additional depths and normal information

    Args:
        image_idx: specific image index to work with
        semantics: semantics data
    """

    # depth
    depth = depths[image_idx]
    # normal
    normal = normals[image_idx]

    return {"depth": depth, "normal": normal}


def get_sensor_depths(image_idx: int, sensor_depths):
    """function to process additional sensor depths

    Args:
        image_idx: specific image index to work with
        sensor_depths: semantics data
    """

    # sensor depth
    sensor_depth = sensor_depths[image_idx]

    return {"sensor_depth": sensor_depth}


def get_foreground_masks(image_idx: int, fg_masks):
    """function to process additional foreground_masks

    Args:
        image_idx: specific image index to work with
        fg_masks: foreground_masks
    """

    # sensor depth
    fg_mask = fg_masks[image_idx]

    return {"fg_mask": fg_mask}


@dataclass
class LumaDataParserConfig(DataParserConfig):
    """Scene dataset parser config"""

    _target: Type = field(default_factory=lambda: Luma)
    """target class to instantiate"""
    data: Path = Path("data/DTU/scan65")
    """Directory specifying location of data."""
    include_mono_prior: bool = False
    """whether or not to load monocular depth and normal """
    include_sensor_depth: bool = False
    """whether or not to load sensor depth"""
    include_foreground_mask: bool = False
    """whether or not to load foreground mask"""
    include_sfm_points: bool = False
    """whether or not to load sfm points"""
    downscale_factor: int = 1
    scene_scale: float = 2.0
    """
    Sets the bounding cube to have edge length of this size.
    The longest dimension of the Friends axis-aligned bbox will be scaled to this value.
    """
    load_pairs: bool = False
    """whether to load pairs for multi-view consistency"""
    neighbors_num: Optional[int] = None
    neighbors_shuffle: Optional[bool] = False
    pairs_sorted_ascending: Optional[bool] = True
    """if src image pairs are sorted in ascending order by similarity i.e. 
    the last element is the most similar to the first (ref)"""
    skip_every_for_val_split: int = 1
    """sub sampling validation images"""
    train_val_no_overlap: bool = False
    """remove selected / sampled validation images from training set"""
    auto_orient: bool = False


@dataclass
class Luma(DataParser):
    """Luma Dataset"""

    config: LumaDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):  # pylint: disable=unused-argument,too-many-statements
        # load meta data
        intrinsics_fp = os.path.join(self.config.data, "intrinsics.json")
        images_dir = os.path.join(self.config.data, "images")
        pose_dir = os.path.join(self.config.data, "pose")

        intrinsics_data = load_from_json(self.config.data / "intrinsics.json")
        intrinsics = torch.Tensor(
            [
                [intrinsics_data["fx"], 0, intrinsics_data["cx"], 0],
                [0, intrinsics_data["fy"], intrinsics_data["cy"], 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        )
        # meta = load_from_json(self.config.data / "meta_data.json")
        # indices = list(range(len(meta["frames"])))
        indices = list(map(lambda x : x.split('.')[0], os.listdir(images_dir)))

        # subsample to avoid out-of-memory for validation set
        if split != "train" and self.config.skip_every_for_val_split >= 1:
            indices = indices[:: self.config.skip_every_for_val_split]
        else:
            # if you use this option, training set should not contain any image in validation set
            if self.config.train_val_no_overlap:
                indices = [i for i in indices if i % self.config.skip_every_for_val_split != 0]

        image_filenames = []
        depth_images = []
        normal_images = []
        sensor_depth_images = []
        foreground_mask_images = []
        sfm_points = []
        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []
        for i in indices:

            image_filename = self.config.data / "images" / (i + ".png")
            # append data
            image_filenames.append(image_filename)

            camtoworld_str = ""
            with open(self.config.data / "pose" / (i + ".txt"), "r") as f:
                camtoworld_str = f.read()
            camtoworld_list = list(map(
                lambda x : list(map(lambda y : float(y), x.split(" "))),
                camtoworld_str.split("\n")[:4]
            ))
            camtoworld = torch.tensor(camtoworld_list)
            camera_to_worlds.append(camtoworld)

            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])

            assert self.config.include_mono_prior == False
            assert self.config.include_sensor_depth == False
            assert self.config.include_foreground_mask == False
            assert self.config.include_sfm_points == False

        fx = torch.stack(fx)
        fy = torch.stack(fy)
        cx = torch.stack(cx)
        cy = torch.stack(cy)
        camera_to_worlds = torch.stack(camera_to_worlds)

        # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
        camera_to_worlds[:, 0:3, 1:3] *= -1

        if self.config.auto_orient:
            camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
                camera_to_worlds,
                method="up",
                center_poses=False,
            )

            # we should also transform normal accordingly
            normal_images_aligned = []
            for normal_image in normal_images:
                h, w, _ = normal_image.shape
                normal_image = transform[:3, :3] @ normal_image.reshape(-1, 3).permute(1, 0)
                normal_image = normal_image.permute(1, 0).reshape(h, w, 3)
                normal_images_aligned.append(normal_image)
            normal_images = normal_images_aligned

        points = np.load(self.config.data / "points.npy")

        print(points.shape, "hey")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud("test_pcd.ply", pcd)
        # scene box from meta data
        aabb = torch.tensor([
            np.min(points, axis=0),
            np.max(points, axis=0)
        ], dtype=torch.float32)
        radius = torch.norm(aabb[1]-aabb[0]).item()
        print("radius", radius)
        scene_box = SceneBox(
            aabb=aabb,
        )

        print("AABB", aabb)

        height, width =  intrinsics_data["extraction_image_height"], intrinsics_data["extraction_image_width"]
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        # TODO supports downsample
        # cameras.rescale_output_resolution(scaling_factor=1.0 / self.config.downscale_factor)

        additional_inputs_dict = {}

        # load pair information
        # pairs_path = self.config.data / "pairs.txt"
        if split == "train" and self.config.load_pairs:
            print('loading_pairs')
            matches = np.load(self.config.data / "matches.npz")
            print("matches", matches.files)
            print("matches", matches['001562'].shape, matches['001562'])
            pairs_srcs = []
            for file_name in matches.files:
                print(file_name, list(map(int, matches[file_name])))
                sources_array = [int(file_name)] + list(map(int, matches[file_name]))
                # if self.config.pairs_sorted_ascending:
                # invert (flip) the source elements s.t. the most similar source is in index 1 (index 0 is reference)
                # sources_array = [sources_array[0]] + sources_array[:1:-1]
                pairs_srcs.append(sources_array)
            pairs_srcs = torch.tensor(pairs_srcs)
            all_imgs = torch.stack(
                [get_image(image_filename) for image_filename in sorted(image_filenames)], axis=0
            ).cuda()

            additional_inputs_dict["pairs"] = {
                "func": get_src_from_pairs,
                "kwargs": {
                    "all_imgs": all_imgs,
                    "pairs_srcs": pairs_srcs,
                    "neighbors_num": self.config.neighbors_num,
                    "neighbors_shuffle": self.config.neighbors_shuffle,
                },
            }

        metadata = {
            "depth_filenames": [],
            "normal_filenames": [],
            "camera_to_worlds": [],
            "transform": [],
            "include_mono_prior": False,
        }

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
            # additional_inputs=additional_inputs_dict,
            # depths=depth_images,
            # normals=normal_images,
        )
        return dataparser_outputs
