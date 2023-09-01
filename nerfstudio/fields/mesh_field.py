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
import os

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import nvdiffrast.torch as dr
import open3d as o3d

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle, RaySamples

import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from nerfstudio.exporter.exporter_utils import generate_point_cloud

from nerfstudio.fields.base_field import Field

from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.custom_eval_utils import eval_setup
from nerfstudio.utils.timer import Timer

import nerfstudio.utils.poses as pose_utils
from nerfstudio.viewer.server.viewer_elements import ViewerControl, ViewerButton

import torch_scatter

from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
)

from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.nero_field import NeROField, NeROFieldConfig
from nerfstudio.models.nero import NeROModelConfig

from torch.profiler import profile, record_function, ProfilerActivity


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


class SingleCamera:
    def __init__(self, camera, idx):
        self.camera = camera.to(torch.device("cuda"))
        self.idx = idx

        self.width = self.camera.width
        self.height = self.camera.height

        near = 0.01
        far = 1000
        mvp = pose_utils.inverse(self.camera.camera_to_worlds)
        y = self.camera.height / (2.0 * self.camera.fy)
        aspect = self.camera.width / self.camera.height

        projection = torch.tensor(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],
                [0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near)],
                [0, 0, -1, 0],
            ],
            dtype=torch.float,
            device=self.camera.device,
        )
        # projection = torch.from_numpy(projection).cuda()

        self.mvp = projection @ pose_utils.to4x4(mvp)

    def generate_rays(self, timer):
        timer.start("RAY-A1")
        rays = self.camera.generate_rays(camera_indices=0)
        timer.stop()
        print("RAYS", self.camera.device, rays.origins.device)
        timer.start("RAY-A2")
        rays = rays.to(self.mvp.device)
        timer.stop()
        return rays


class Mesh:
    def __init__(self, vertices, faces, rgbs, timer):
        self.vertices = vertices
        self.faces = faces
        # self.faces = torch.flip(faces, dims=(1,))
        self.rgbs = rgbs
        self.compute_normals()
        self.timer = timer

    def compute_normals(self):
        ba = self.vertices[self.faces[:, 1], :] - self.vertices[self.faces[:, 0], :]
        ca = self.vertices[self.faces[:, 2], :] - self.vertices[self.faces[:, 0], :]
        baxca = torch.linalg.cross(ba, ca)
        areas = 0.5 * torch.linalg.norm(baxca, dim=-1)
        self.face_normals = baxca / torch.clamp(2.0 * areas[:, None], min=1e-20)

        # check_face_normals = torch.linalg.norm(self.face_normals, dim=-1)
        # print("check face normals", check_face_normals.shape, check_face_normals.min().item(), check_face_normals.max().item(), check_face_normals.mean().item())

        self.vertex_normals = torch.zeros_like(self.vertices)
        self.vertex_areas = torch.zeros_like(self.vertices[:, 0])
        vertex_idx = self.faces.long()
        for i in range(3):
            for j in range(3):
                torch_scatter.scatter_add(baxca[:, j], vertex_idx[:, i], out=self.vertex_normals[:, j])
            torch_scatter.scatter_add(areas, vertex_idx[:, i], out=self.vertex_areas)

        self.vertex_normals /= torch.clamp(2.0 * self.vertex_areas[:, None], min=1e-20)
        # check_vertex_normals = torch.linalg.norm(self.vertex_normals, dim=-1)
        # print("check vertex normals", check_vertex_normals.shape, check_vertex_normals.min().item(), check_vertex_normals.max().item(), check_vertex_normals.mean().item())
        self.vertex_normals = safe_normalize(self.vertex_normals)

    def laplacian_cot(self):
        """
        Compute the cotangent laplacian
        Inspired by https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_laplacian_smoothing.html
        Parameters
        ----------
        verts : torch.Tensor
            Vertex positions.
        faces : torch.Tensor
            array of triangle faces.
        """

        # V = sum(V_n), F = sum(F_n)
        verts = self.vertices.float()
        faces = self.faces.long()
        V, F = verts.shape[0], faces.shape[0]

        face_verts = verts[faces]
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

        # Side lengths of each triangle, of shape (sum(F_n),)
        # A is the side opposite v1, B is opposite v2, and C is opposite v3
        A = (v1 - v2).norm(dim=1)
        B = (v0 - v2).norm(dim=1)
        C = (v0 - v1).norm(dim=1)

        # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
        s = 0.5 * (A + B + C)
        # note that the area can be negative (close to 0) causing nans after sqrt()
        # we clip it to a small positive value
        area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

        # Compute cotangents of angles, of shape (sum(F_n), 3)
        A2, B2, C2 = A * A, B * B, C * C
        cota = (B2 + C2 - A2) / area
        cotb = (A2 + C2 - B2) / area
        cotc = (A2 + B2 - C2) / area
        cot = torch.stack([cota, cotb, cotc], dim=1)
        cot /= 4.0

        # Construct a sparse matrix by basically doing:
        # L[v1, v2] = cota
        # L[v2, v0] = cotb
        # L[v0, v1] = cotc
        ii = faces[:, [1, 2, 0]]
        jj = faces[:, [2, 0, 1]]
        idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
        L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

        # Make it symmetric; this means we are also setting
        # L[v2, v1] = cota
        # L[v0, v2] = cotb
        # L[v1, v0] = cotc
        L += L.t()

        # Add the diagonal indices
        vals = torch.sparse.sum(L, dim=0).to_dense()
        indices = torch.arange(V, device="cuda")
        idx = torch.stack([indices, indices], dim=0)
        L = torch.sparse.FloatTensor(idx, vals, (V, V)) - L
        return L

    def laplacian_uniform(self):
        """
        Compute the uniform laplacian
        Parameters
        ----------
        verts : torch.Tensor
            Vertex positions.
        faces : torch.Tensor
            array of triangle faces.
        """
        verts = self.vertices.float()
        faces = self.faces.long()
        V = verts.shape[0]
        F = faces.shape[0]

        # Neighbor indices
        ii = faces[:, [1, 2, 0]].flatten()
        jj = faces[:, [2, 0, 1]].flatten()
        adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
        adj_values = torch.ones(adj.shape[1], device=verts.device, dtype=torch.float)

        # Diagonal indices
        diag_idx = adj[0]

        # Build the sparse matrix
        idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
        values = torch.cat((-adj_values, adj_values))

        # The coalesce operation sums the duplicate indices, resulting in the
        # correct diagonal
        return torch.sparse_coo_tensor(idx, values, (V, V)).coalesce()

    def mark_faces(self, ids):
        marked_mesh = o3d.geometry.TriangleMesh()
        marked_mesh.vertices = o3d.utility.Vector3dVector(self.vertices.detach().cpu().numpy())
        marked_mesh.triangles = o3d.utility.Vector3iVector(self.faces.detach().cpu().numpy())
        vertex_colors = torch.zeros_like(self.vertices)
        vertex_colors[self.faces[ids, 0], 0] = 1
        vertex_colors[self.faces[ids, 1], 0] = 1
        vertex_colors[self.faces[ids, 2], 0] = 1
        marked_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors.detach().cpu().numpy())
        o3d.io.write_triangle_mesh("temp/marked_mesh.obj", marked_mesh)

    def render(self, glctx, camera, rgb_fn, config, save_filename=None, **kwargs):
        self.timer.lap()
        self.timer.start("A1")
        rays = camera.generate_rays(self.timer)
        self.timer.stop()
        self.timer.start("A2")
        rays_o = rays.origins
        rays_d = rays.directions
        h0 = camera.height
        w0 = camera.width
        mvp = camera.mvp

        h8 = ((h0 + 7) // 8) * 8
        w8 = ((w0 + 7) // 8) * 8
        h, w = h0, w0
        h4 = (h8 - h0) // 2
        w4 = (w8 - w0) // 2

        dirs = safe_normalize(rays_d)
        self.timer.stop()
        bg_color = 1

        results = {}
        print(self.vertices.device, mvp.device, "DEVICE")
        self.timer.start("B1")
        vertices_clip = (
            torch.matmul(F.pad(self.vertices, pad=(0, 1), mode="constant", value=1.0), torch.transpose(mvp, 0, 1))
            .float()
            .unsqueeze(0)
        )  # [1, N, 4]
        self.timer.stop()
        self.timer.start("B2")
        rast, _ = dr.rasterize(glctx, vertices_clip, self.faces, (h8, w8))
        self.timer.stop()
        # self.mark_faces(torch.flatten(rast[...,-1]).to(torch.int32)-1)
        self.timer.start("C1")
        xyzs = dr.interpolate(self.vertices.unsqueeze(0), rast, self.faces)[0]  # [1, H, W, 3]
        rgbs = dr.interpolate(self.rgbs.unsqueeze(0), rast, self.faces)[0]
        normals = dr.interpolate(self.vertex_normals.unsqueeze(0), rast, self.faces)[0]
        mask = dr.interpolate(torch.ones_like(self.vertices[:, :1]).unsqueeze(0), rast, self.faces)[0]  # [1, H, W, 1]
        mask_flatten = (mask > 0).view(-1).detach()
        self.timer.stop()

        self.timer.start("D1")
        keys = [
            # 'reflective',
            # 'occ_prob',
            "diffuse_color",
            # 'diffuse_light',
            "specular_color",
            "specular_ref",
            "specular_light",
            # 'indirect_light',
            "specular_albedo",
            "diffuse_albedo",
            "metallic",
        ]
        if config.shading_mode != "nero":
            keys = []
        extra_info = {key: rgbs.clone() for key in keys}
        self.timer.stop()

        if config.use_rgb_model:
            self.timer.start("D2")
            diff = xyzs[0, h4 : h0 + h4, w4 : w0 + w4, :] - rays_o
            dis = torch.sqrt(
                torch.sum(diff * diff, dim=-1)
                / torch.clamp(torch.sum(rays.directions * rays.directions, dim=-1), min=1e-20)
            )
            print("DIS", dis.mean(), "WTF")
            eps = 1e-10
            ray_samples = rays.get_ray_samples(dis[:, :, None, None] - eps, dis[:, :, None, None] + eps)
            # print("RAY SAMPLES", ray_samples.shape, ray_samples[0].shape, normal_samples.shape, normal_samples[0].shape)
            self.timer.stop()
            if config.shading_mode == "nero":
                self.timer.start("D3")
                ray_samples = ray_samples.reshape((-1, 1))
                normal_samples = normals[0, h4 : h0 + h4, w4 : w0 + w4, :]
                normal_samples = normal_samples.reshape((-1, 3))
                new_rgbs = torch.zeros_like(normal_samples)

                new_extra_info = {key: torch.zeros_like(normal_samples) for key in keys}

                print("RAY SAMPLES", ray_samples.shape, normal_samples.shape)
                BS = ray_samples.shape[0]  # 262144
                self.timer.stop()
                for i in range(0, ray_samples.shape[0], BS):
                    l = i
                    r = min(l + BS, ray_samples.shape[0])
                    # with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
                    # with torch.cuda.amp.autocast(enabled=False):
                    self.timer.start("D4")
                    outputs = rgb_fn(ray_samples[l:r], normal_samples[l:r])
                    self.timer.stop()
                    self.timer.start("D5")
                    new_rgbs[l:r] = outputs[FieldHeadNames.RGB][:, 0, :]
                    self.timer.stop()
                    for key in keys:
                        self.timer.start("D6")
                        new_extra_info[key][l:r] = outputs[key][:, 0, :]
                        self.timer.stop()
                        # print("NEW RGBS", new_rgbs[l:r].min().item(), new_rgbs[l:r].max().item())
                    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

                    # print(i, l, r, ray_samples.shape, new_rgbs.mean(dim=0))
                    self.timer.start("D7")
                    # torch.cuda.empty_cache()
                    self.timer.stop()
                self.timer.start("D8")
                new_rgbs = new_rgbs.reshape((h0, w0, 1, 3))
                self.timer.stop()
                for key in keys:
                    self.timer.start("D9")
                    new_extra_info[key] = new_extra_info[key].reshape((h0, w0, 1, 3))
                    self.timer.stop()

            else:
                new_rgbs = rgb_fn(ray_samples)[FieldHeadNames.RGB]
            self.timer.start("E1")
            rgbs[0, h4 : h0 + h4, w4 : w0 + w4, :] = new_rgbs[:, :, 0, :]
            self.timer.stop()
            for key in keys:
                self.timer.start("E2")
                extra_info[key][0, h4 : h0 + h4, w4 : w0 + w4, :] = new_extra_info[key][:, :, 0, :]
                self.timer.stop()

        if config.shading_mode == "lambertian":
            normals = safe_normalize(normals)
            lambertian = (-dirs * normals[0, h4 : h0 + h4, w4 : w0 + w4, :]).sum(dim=-1)

            check_lambertian = lambertian[mask[0, :, :, 0] > 0]
            print(
                "lambertian",
                check_lambertian.min().item(),
                check_lambertian.max().item(),
                check_lambertian.mean().item(),
            )
            rgbs[:, h4 : h0 + h4, w4 : w0 + w4, :] *= lambertian[None, :, :, None]

        alphas = mask.float()

        # print("RGBS", self.rgbs.min(), self.rgbs.max(), rgbs.min(), rgbs.max())
        self.timer.start("F1")
        alphas = dr.antialias(alphas, rast, vertices_clip, self.faces, pos_gradient_boost=True).squeeze(0).clamp(0, 1)
        rgbs = dr.antialias(rgbs, rast, vertices_clip, self.faces, pos_gradient_boost=True).squeeze(0).clamp(0, 1)
        self.timer.stop()
        for key in keys:
            self.timer.start("F2")
            extra_info[key] = dr.antialias(
                extra_info[key], rast, vertices_clip, self.faces, pos_gradient_boost=True
            ).squeeze(0)
            self.timer.stop()

        self.timer.start("G1")
        image = alphas * rgbs
        depth = alphas * rast[0, :, :, [2]]
        T = 1 - alphas

        # trig_id for updating trig errors
        trig_id = rast[0, :, :, -1] - 1  # [h, w]

        # ssaa
        # if self.opt.ssaa > 1:
        #     image = scale_img_hwc(image, (h0, w0))
        #     depth = scale_img_hwc(depth, (h0, w0))
        #     T = scale_img_hwc(T, (h0, w0))
        #     trig_id = scale_img_hw(trig_id.float(), (h0, w0), mag='nearest', min='nearest')

        self.triangles_errors_id = trig_id

        image = image + T * bg_color

        self.timer.stop()
        # if save_filename is not None and random.randint(0, 1000000) % 100 == 0:
        if "32" in save_filename:
            save_dir = os.path.join(config.save_dir, "temp", "mesh")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.timer.start("H1")
            plt.matshow(image.detach().cpu().numpy())
            plt.savefig(os.path.join(save_dir, f"{save_filename}.png"))
            plt.close(plt.gcf())
            alphas = alphas.detach().cpu().numpy()
            self.timer.stop()
            if random.randint(0, 1000000) % 10 == 0:
                for key in keys:
                    self.timer.start("H2")
                    data_img = extra_info[key].detach().cpu().numpy()
                    data_img -= np.min(data_img)
                    data_img /= np.max(data_img)
                    data_img *= alphas
                    plt.matshow(data_img)
                    plt.savefig(os.path.join(save_dir, f"{save_filename}_{key}.png"))
                    plt.close(plt.gcf())
                    self.timer.stop()
            self.timer.start("H3")
            plt.matshow((0.5 * (normals[0, ...] + 1)).detach().cpu().numpy())
            plt.savefig(os.path.join(save_dir, f"{save_filename}_normal.png"))
            plt.close(plt.gcf())
            self.timer.stop()

        self.timer.start("I1")
        image = image.view(h8, w8, 3)
        depth = depth.view(h8, w8)

        results["depth"] = depth[h4 : h0 + h4, w4 : w0 + w4].reshape(-1, 1)
        results["rgb"] = image[h4 : h0 + h4, w4 : w0 + w4, :].reshape(-1, 3)
        results["weights_sum"] = (1 - T)[h4 : h0 + h4, w4 : w0 + w4, :].reshape(-1, 1)
        results["dim"] = (h0, w0)
        self.timer.stop()

        self.timer.dump()

        return results


def get_disconnected_mesh(mesh):
    vertices, faces, rgbs = mesh.vertices, mesh.faces, mesh.rgbs

    new_vertices = vertices[faces.reshape(-1), :]
    new_rgbs = rgbs[faces.reshape(-1), :]
    new_faces = (
        torch.arange(start=0, end=faces.shape[0] * faces.shape[1], dtype=torch.int32)
        .reshape(faces.shape[0], faces.shape[1])
        .cuda()
    )

    return Mesh(new_vertices, new_faces, new_rgbs)


def surround_mesh(mesh):
    vertices = np.array(mesh.vertices)
    aabb = [np.min(vertices, axis=0), np.max(vertices, axis=0)]
    center = 0.5 * (aabb[1] + aabb[0])
    extent = 0.5 * (aabb[1] - aabb[0])
    corners = np.array(
        [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [-1, 1, 1],
            [-1, -1, 1],
            [-1, 1, -1],
            [1, -1, -1],
            [-1, -1, -1],
        ]
    )
    indices = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [0, 1, 3],
            [1, 2, 6],
            [2, 3, 4],
            [3, 1, 5],
            [7, 4, 5],
            [7, 5, 6],
            [7, 6, 4],
            [5, 4, 3],
            [6, 5, 1],
            [4, 6, 2],
        ]
    )
    aabb_mesh = o3d.geometry.TriangleMesh()
    aabb_mesh.vertices = o3d.utility.Vector3dVector(center + 20.0 * extent * corners)
    aabb_mesh.triangles = o3d.utility.Vector3iVector(indices[:, ::-1])
    aabb_mesh = aabb_mesh.subdivide_loop(number_of_iterations=6)
    aabb_vertices = np.array(aabb_mesh.vertices)
    aabb_mesh.vertex_colors = o3d.utility.Vector3dVector(0.5 * np.ones_like(aabb_vertices))
    # o3d.io.write_triangle_mesh("test_aabb.obj", aabb_mesh)
    return aabb_mesh


def merge_meshes(meshes):
    vertices = []
    faces = []
    rgbs = []
    vertex_cnt = 0
    for i in range(len(meshes)):
        vertices.append(np.array(meshes[i].vertices))
        faces.append(np.array(meshes[i].triangles) + vertex_cnt)
        rgb = np.array(meshes[i].vertex_colors)
        if rgb.size == 0:
            rgb = 0.5 * np.ones_like(vertices[i])
        rgbs.append(rgb)
        vertex_cnt += vertices[i].shape[0]

    vertices = np.concatenate(vertices, axis=0)
    faces = np.concatenate(faces, axis=0)
    rgbs = np.concatenate(rgbs, axis=0)

    merged_mesh = o3d.geometry.TriangleMesh()
    merged_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    merged_mesh.triangles = o3d.utility.Vector3iVector(faces)
    merged_mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs)
    # o3d.io.write_triangle_mesh("test_merged.obj", merged_mesh)
    return merged_mesh


class MeshField(Field):
    def __init__(self, mesh, config, rgb_model=None, timer=None):
        super().__init__()
        self.timer = timer
        if timer is None:
            self.timer = Timer()

        # torch.autograd.set_detect_anomaly(True)
        # torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        self.config = config
        # bg_mesh = surround_mesh(mesh)
        # merged_mesh = merge_meshes([mesh, bg_mesh])
        merged_mesh = mesh

        # self.glctx = dr.RasterizeGLContext(output_db=False)
        self.glctx = dr.RasterizeCudaContext()

        self.vertices = torch.from_numpy(np.array(merged_mesh.vertices)).float().cuda()
        self.faces = torch.from_numpy(np.array(merged_mesh.triangles)).cuda()

        if config.shading_mode == "nero":
            # nero_config = NeROFieldConfig(
            #     inside_outside=False,
            # )
            # self.rgb_model = NeROField(
            #     config=nero_config,
            #     aabb=rgb_model.aabb,
            #     spatial_distortion=SceneContraction(order=float("inf")),
            #     num_images=240,
            #     use_average_appearance_embedding=False,
            # )

            self.rgb_model = NeROField(
                config.aabb,
                hidden_dim=64,
                num_levels=2,
                max_res=2048,
                log2_hashmap_size=19,
                hidden_dim_color=64,
                hidden_dim_transient=64,
                spatial_distortion=SceneContraction(order=float("inf")),
                num_images=config.num_train_data,
                use_pred_normals=False,
                use_average_appearance_embedding=True,
                appearance_embedding_dim=32,
                geo_feat_dim=config.geo_feat_dim,
                use_weight_norm=config.weight_norm,
                hash_encoding=config.hash_encoding,
                implementation="torch",
                initial_model=None,
                # initial_model=rgb_model
                timer=self.timer,
            )
            self.rgb_fn = lambda ray_samples, normal_samples: self.rgb_model.get_special_outputs(
                ray_samples, normal_samples
            )
        else:
            self.rgb_model = rgb_model
            self.rgb_fn = lambda ray_samples: rgb_model(ray_samples)

        # self.rgbs = None
        # if self.rgb_model is None:
        self.rgbs = nn.Parameter(
            torch.from_numpy(np.array(merged_mesh.vertex_colors)).float().cuda(), requires_grad=True
        )
        # self.rgbs = torch.from_numpy(np.array(merged_mesh.vertex_colors)).float().cuda()
        self.vertex_offsets = nn.Parameter(torch.zeros_like(self.vertices), requires_grad=True)
        # self.vertex_offsets = torch.zeros_like(self.vertices)

    def get_camera_outputs(self, camera: SingleCamera, is_viewer: bool, save_filename=None):
        self.timer.start("MSH-A1")
        mesh = Mesh(self.vertices + self.vertex_offsets, self.faces, self.rgbs, self.timer)
        self.cache_mesh = mesh
        self.timer.stop()
        # face_mesh = get_disconnected_mesh(mesh)
        outputs = mesh.render(
            self.glctx,
            camera,
            self.rgb_fn,
            self.config,
            save_filename=save_filename,
        )
        return outputs

    def get_bundle_outputs(self, ray_bundle: RayBundle, camera: SingleCamera, is_viewer: bool):
        idx = camera.idx
        # print("LEARNING", self.vertex_offsets.abs().mean().item(), self.rgbs.abs().mean().item())
        save_filename = str(idx) if not is_viewer else None
        return self.get_camera_outputs(camera, is_viewer, save_filename)

    def vertex_offset_loss(self):
        return torch.linalg.norm(self.vertex_offsets).mean()

    def laplacian_smooth_loss(self):
        self.timer.start("LAP-A1")
        verts = (self.vertices + self.vertex_offsets).float()
        faces = self.faces.long()
        cnt_vert = torch.zeros_like(verts[:, 0])
        avg_vert = torch.zeros_like(verts)
        for i in range(3):
            for j in range(3):
                ii = (i + 1) % 3
                iii = (ii + 1) % 3
                torch_scatter.scatter_add(verts[faces[:, ii], j], faces[:, i], out=avg_vert[:, j])
                torch_scatter.scatter_add(verts[faces[:, iii], j], faces[:, i], out=avg_vert[:, j])
            torch_scatter.scatter_add(2.0 * torch.ones_like(faces[:, i]).float(), faces[:, i], out=cnt_vert)

        mask = cnt_vert > 0.5
        avg_vert[mask] /= cnt_vert[mask, None]
        loss = (verts - avg_vert)[mask]

        loss = loss.norm(dim=1)
        loss = loss.mean()
        self.timer.stop()
        return loss

    def laplacian_smooth_loss_broken(self, cotan=False):
        verts = self.cache_mesh.vertices.float()
        with torch.no_grad():
            if cotan:
                L = self.cache_mesh.laplacian_cot()
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                mask = norm_w > 0
                norm_w[mask] = 1.0 / norm_w[mask]
            else:
                L = self.cache_mesh.laplacian_uniform()
        with torch.autocast("cuda"):
            if cotan:
                loss = L.mm(verts) * norm_w - verts
            else:
                loss = L.mm(verts)
        # loss = loss.abs().sum(dim=1)
        loss = loss.norm(dim=1)
        loss = loss.mean()
        return loss

    def rgb_variation_loss(self):
        loss = 0
        for i in range(3):
            j = (i + 1) % 3
            rgb_var = torch.linalg.norm(
                (
                    self.cache_mesh.rgbs[self.cache_mesh.faces[:, i], :]
                    - self.cache_mesh.rgbs[self.cache_mesh.faces[:, j], :]
                )
            )
            vert_dis = torch.linalg.norm(
                (
                    self.cache_mesh.vertices[self.cache_mesh.faces[:, i], :]
                    - self.cache_mesh.vertices[self.cache_mesh.faces[:, j], :]
                )
            )
            loss += (torch.exp(-400.0 * vert_dis) * rgb_var).mean()
        return loss

    def save_mesh(self, path: Path):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector((self.vertices + self.vertex_offsets).detach().cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(self.faces.detach().cpu().numpy())
        mesh.vertex_colors = o3d.utility.Vector3dVector(self.rgbs.detach().cpu().numpy())
        print("STR", str(path))
        o3d.io.write_triangle_mesh(str(path), mesh)
