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

from typing import Callable, List, Optional, Tuple, Union

import random
import os
import math
import numpy as np
import torch
import trimesh
from jaxtyping import Bool, Float
from skimage import measure
from torch import Tensor

from tqdm import tqdm
import time

import open3d as o3d
from matplotlib import colormaps

from nerfstudio.utils.timer import Timer

SUB_COEF = torch.IntTensor(
    [
        [[0, 0, 0], [1, 1, 1]],
        [[1, 0, 0], [2, 1, 1]],
        [[0, 1, 0], [1, 2, 1]],
        [[0, 0, 1], [1, 1, 2]],
        [[0, 1, 1], [1, 2, 2]],
        [[1, 0, 1], [2, 1, 2]],
        [[1, 1, 0], [2, 2, 1]],
        [[1, 1, 1], [2, 2, 2]],
    ]
).cpu()

EDGE_COEF = torch.IntTensor(
    [
        [[0, 0, 0], [1, 0, 0]],
        [[0, 0, 0], [0, 1, 0]],
        [[0, 0, 0], [0, 0, 1]],
        [[1, 0, 0], [1, 1, 0]],
        [[1, 0, 0], [1, 0, 1]],
        [[0, 1, 0], [1, 1, 0]],
        [[0, 1, 0], [0, 1, 1]],
        [[0, 0, 1], [1, 0, 1]],
        [[0, 0, 1], [0, 1, 1]],
        [[0, 1, 1], [1, 1, 1]],
        [[1, 0, 1], [1, 1, 1]],
        [[1, 1, 0], [1, 1, 1]],
    ]
).cpu()

EDGE_IDX = [0, 1, 2, 1, 2, 0, 2, 0, 1, 0, 1, 2]


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def mask_points_with_aabb(points: Tensor, aabb: Tensor) -> Tensor:
    mask_xyz = [(aabb[0][..., i] <= points[..., i]) & (points[..., i] <= aabb[1][..., i]) for i in range(3)]
    return mask_xyz[0] & mask_xyz[1] & mask_xyz[2]


def scale_in_aabb(aabb, int_pos, scale):
    weights = int_pos / scale
    return weights * aabb[1:, :] + (1 - weights) * aabb[:1, :]


def subdivide_aabb(aabb):
    split_aabbs = (SUB_COEF * aabb[None, 1:, :] + (2 - SUB_COEF) * aabb[None, :1, :]) // 2
    return split_aabbs


def get_aabb_edges(aabb, depth):
    edge_vertices = EDGE_COEF * aabb[None, 1:, :] + (1 - EDGE_COEF) * aabb[None, :1, :]
    edges = [
        OctreeEdge(
            edge_vertices[j, 0, ...],
            edge_vertices[j, 1, ...],
            EDGE_IDX[j],
            depth,
        )
        for j in range(12)
    ]
    return edges


class Plane:
    def __init__(self, normal, constant, point=None):
        self.normal = normal
        self.constant = constant
        self.point = point


class OctreeGlobal:
    def __init__(self, aabb, min_depth, max_depth, points, timer):
        self.aabb = aabb
        self.id_counter = 0
        self.edges = []
        self.centers = []

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.points = points
        self.octree = []
        self.timer = timer

        self.center_map = {}
        self.used_edge = {}

    def build_octree_iterative(self):
        int_aabb = torch.stack(
            [torch.zeros((3,), dtype=torch.int32), torch.ones((3,), dtype=torch.int32) * (1 << (self.max_depth + 1))]
        ).cpu()
        root = OctreeNode(int_aabb, 0, self, points=self.points, min_depth=self.min_depth, max_depth=self.max_depth)
        self.octree.append(root)
        points_hierarchy = [None for i in range(self.max_depth + 2)]
        points_hierarchy[0] = self.points
        prev_depth = 0

        points = points_hierarchy[prev_depth]

        node_stack = [0]
        cnt = 0
        while len(node_stack) > 0:
            if cnt % 100000 == 0:
                print("count", cnt)
            cnt += 1

            i = node_stack[-1]
            node_stack.pop()

            self.timer.start("1")
            cur_depth = self.octree[i].depth
            if cur_depth <= prev_depth:
                points = points_hierarchy[cur_depth]
            self.timer.stop()
            self.timer.start("2")
            if points is None:
                mask = None
            else:
                mask = mask_points_with_aabb(points, self.octree[i].aabb)
                points = points[mask, :]
                points_hierarchy[cur_depth + 1] = points
            self.timer.stop()
            self.timer.start("3")
            check_condition = self.octree[i].subdivide_condition(mask)
            self.timer.stop()
            # print(mask.sum().item(), cur_depth, "CURRENT", check_condition, self.octree[i].min_depth, self.octree[i].max_depth)

            if check_condition:
                self.timer.start("4a")
                self.octree += self.octree[i].subdivide(points)
                node_stack += [self.id_counter - 1 - j for j in range(8)]
                self.timer.stop()
            else:
                self.timer.start("4b")
                self.edges += self.octree[i].edges
                self.timer.stop()
            prev_depth = cur_depth

    def pop_count(self):
        cur = self.id_counter
        self.id_counter += 1
        return cur


class OctreeEdge:
    def __init__(self, point1: Tensor, point2: Tensor, axis: int, depth: int):
        self.point1 = point1
        self.point2 = point2
        self.axis = axis  # x=0, y=1, z=2
        self.depth = depth
        self.constant = (self.point1[(axis + 1) % 3].item(), self.point1[(axis + 2) % 3].item())

    def __lt__(self, other):
        if self.point1[axis] < other.point1[axis]:
            return True
        if self.point1[axis] > other.point1[axis]:
            return False
        if self.point2[axis] < other.point2[axis]:
            return True
        if self.point2[axis] > other.point2[axis]:
            return False
        return True

    def __hash__(self):
        return hash(tuple(self.point1.tolist()) + (self.axis, self.depth))

    def __eq__(self, other):
        return self.depth == other.depth and self.axis == other.axis and (self.point1 == other.point1).all()

    def get_cells(self, og: OctreeGlobal):
        dirs = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
        axis = self.axis
        axis1 = (axis + 1) % 3
        axis2 = (axis1 + 1) % 3
        max_depth = og.max_depth
        cell_ids = []
        cell_depths = []
        # print((self.point2[axis] - self.point1[axis]).item(), 1 << (max_depth - self.depth + 1), "HEY THAT MAKES SENSE RIGHT")

        for i in range(4):
            cell_center = ((self.point1 + self.point2) // 2).cpu()
            direction = torch.zeros_like(cell_center)

            direction[axis1] = dirs[i][0]
            direction[axis2] = dirs[i][1]

            cell_size = 1 << (max_depth - self.depth)

            cell_center = cell_center + direction * cell_size

            for d in range(max_depth - self.depth, max_depth + 1):
                # print("gah", i, d, self.point1[axis], self.point2[axis], edge_center[axis])
                cell_key = tuple(cell_center.tolist())
                if cell_key in og.center_map:
                    new_id = og.center_map[cell_key]
                    # old_depth = -1
                    # if (new_id, i, axis) in og.used_edge:
                    #     old_depth = og.used_edge[(new_id, i, axis)]
                    # if old_depth <= self.depth:
                    #     og.used_edge[(new_id, i, axis)] = self.depth
                    cell_ids.append(new_id)
                    break
                cell_center = (cell_center - cell_size) | (2 * cell_size)
                cell_size *= 2
        cell_ids = list(set(cell_ids))
        return cell_ids


class OctreeNode:
    def __init__(
        self,
        int_aabb: Tensor,
        depth: int,
        og: OctreeGlobal,
        # parent,
        points: Optional[Tensor] = None,
        min_depth: int = 0,
        max_depth: int = 10,
        recursive: bool = False,
    ):
        """
        Form initial OctreeNode with some given max depth
        """
        og.timer.start("5")
        self.og = og
        # self.parent = parent
        self.depth = depth
        self.int_aabb = int_aabb
        self.id = og.pop_count()
        self.min_depth = min_depth
        self.max_depth = max_depth

        og.timer.stop()
        og.timer.start("6")

        self.aabb = scale_in_aabb(self.og.aabb, self.int_aabb, 1 << (self.max_depth + 1))
        self.center = (self.aabb[0] + self.aabb[1]) / 2
        self.og.centers.append(self.center)
        int_center = (self.int_aabb[0] + self.int_aabb[1]) // 2
        og.center_map[tuple(int_center.tolist())] = self.id

        og.timer.stop()
        og.timer.start("7")

        self.edges = get_aabb_edges(self.int_aabb, self.depth)

        og.timer.stop()
        if recursive:
            mask = None
            if points is not None:
                mask = mask_points_with_aabb(points, self.aabb)
            self.children = []

            if self.subdivide_condition(mask):
                self.children = self.subdivide(points)

    def subdivide_condition(self, mask=None):
        if self.depth >= self.max_depth:
            return False
        if self.depth <= self.min_depth:
            return True
        if mask is not None and mask.sum().item() > 0:
            return True
        return False

    def subdivide(self, points: Optional[Tensor] = None):
        split_aabb = subdivide_aabb(self.int_aabb)
        children = []
        for i in range(8):
            children.append(
                OctreeNode(
                    split_aabb[i],
                    self.depth + 1,
                    self.og,
                    points=points,
                    min_depth=self.min_depth,
                    max_depth=self.max_depth,
                )
            )
        return children


def fit_to_planes(planes, center=None):
    if center is not None:
        weight = 0.1
        for i in range(3):
            normal = torch.zeros_like(center)
            normal[i] = 1
            planes.append(Plane(weight * normal, -weight * center[i]))
    A = torch.zeros((len(planes), 3))
    b = torch.zeros((len(planes), 1))
    # print("START HM", len(planes))
    for i in range(len(planes)):
        A[i, :] = planes[i].normal
        b[i] = -planes[i].constant
        # print(i, A[i,:])
    x = torch.linalg.lstsq(A, b).solution[:, 0]
    return x


def dual_contour(implicit_fn, normal_fn, pcd, aabb, min_depth, max_depth):
    timer = Timer(cuda_support=False)
    timer.lap()
    if pcd is not None:
        pcd = pcd.cpu()
    aabb = aabb.cpu()
    og = OctreeGlobal(aabb, min_depth, max_depth, pcd, timer)

    int_aabb = torch.stack(
        [torch.zeros((3,), dtype=torch.int32), torch.ones((3,), dtype=torch.int32) * (1 << (max_depth + 1))]
    ).cpu()
    og.build_octree_iterative()

    print("DONE BUILDING")

    print("INITIAL PASS")

    timer.start("A1")
    edges = list(set(og.edges))
    edges.sort(key=lambda x: (x.constant, x.axis, x.point1[x.axis].item(), -x.depth))
    timer.stop()

    faces = []

    corner_id = {}
    corners = []

    process_list = []

    print("OBTAINING KEY EDGES")
    for i in tqdm(range(len(edges))):
        edge = edges[i]
        timer.start("B1")
        ax = edge.axis
        ax1 = (ax + 1) % 3
        ax2 = (ax + 2) % 3
        constant = edge.constant
        timer.stop()
        if (
            i != 0
            and constant == edges[i - 1].constant
            and ax == edges[i - 1].axis
            and edge.point1[ax].item() == edges[i - 1].point1[ax].item()
        ):
            continue
        # edge_dict[key].sort()

        # print(ax, constant)
        # events = list(map(lambda x : (x.point1[x.axis].item(), 1, x.id), edge_dict[key])) + \
        #         list(map(lambda x : (x.point2[x.axis].item(), 0, x.id), edge_dict[key]))

        timer.start("B2")
        face = edge.get_cells(og)
        timer.stop()
        if len(face) <= 2:
            continue
        timer.start("B3")
        process_list.append((edge, face))
        timer.stop()

        timer.start("B4")
        endpoints = [tuple(edge.point1.tolist()), tuple(edge.point2.tolist())]
        for corner in endpoints:
            if corner not in corner_id:
                corner_id[corner] = len(corners)
                corners.append(corner)
        timer.stop()

    print("EVALUATING AT CORNERS")
    timer.start("C1")
    corner_positions = scale_in_aabb(og.aabb, torch.Tensor(corners).cpu(), 1 << (max_depth + 1))
    timer.stop()
    timer.start("C2")
    corner_values = implicit_fn(corner_positions.cuda()).cpu()
    timer.stop()
    print(len(corners), corner_values.shape)
    print("DONE EVALUATING AT CORNERS")

    ids = []
    used_ids = {}
    matched_planes = {}

    plane_intermediate = []

    print("PROCESSING KEY EDGES")
    for edge, face in tqdm(process_list):
        timer.start("D1")
        ax = edge.axis
        ax1 = (ax + 1) % 3
        ax2 = (ax + 2) % 3
        constant = edge.constant

        out_endpoints = torch.zeros_like(int_aabb)
        out_endpoints[0] = edge.point1
        out_endpoints[1] = edge.point2
        # out_endpoints[:,ax1] = constant[0]
        # out_endpoints[:,ax2] = constant[1]
        # out_endpoints[0,ax] = min(ep[0] for ep in ax_endpoints)
        # out_endpoints[1,ax] = max(ep[1] for ep in ax_endpoints)

        timer.stop()
        timer.start("D2")
        aabb_endpoints = scale_in_aabb(og.aabb, out_endpoints, 1 << (max_depth + 1))
        timer.stop()
        timer.start("D3")
        fv0 = corner_values[corner_id[tuple(out_endpoints[0].tolist())]].item()
        fv1 = corner_values[corner_id[tuple(out_endpoints[1].tolist())]].item()
        timer.stop()
        if (fv0 < 0 and fv1 < 0) or (fv0 > 0 and fv1 > 0):
            continue

        # in_endpoints = [max(ep[0] for ep in ax_endpoints), min(ep[1] for ep in ax_endpoints)]

        t = fv1 / (fv1 - fv0 + 1e-10)
        # intersection_pos = t * out_endpoints[0,ax].item() + (1-t) * out_endpoints[1,ax].item()
        # if in_endpoints[0] > intersection_pos or in_endpoints[1] < intersection_pos:
        #     continue

        timer.start("D4")
        plane_id = len(plane_intermediate)
        plane_pos = t * aabb_endpoints[0] + (1 - t) * aabb_endpoints[1]
        plane_intermediate.append(plane_pos)
        timer.stop()
        timer.start("D5")
        angles = []
        # print("FACE", len(face))
        for j in range(len(face)):
            if face[j] not in used_ids:
                used_ids[face[j]] = len(ids)
                matched_planes[face[j]] = []
                ids.append(face[j])
            matched_planes[face[j]].append(plane_id)
            x = og.centers[face[j]][ax1] - aabb_endpoints[0][ax1]
            y = og.centers[face[j]][ax2] - aabb_endpoints[0][ax2]
            theta = math.atan2(y, x)
            angles.append(theta)
        _, face = zip(*sorted(zip(angles, face), key=lambda x: x[0]))
        if fv0 > 0:
            face = face[::-1]
        # face += (face[::2] + face[1::2])
        faces.append(face)
        timer.stop()

    print("COMPUTING NORMALS")
    timer.start("E1")
    check_vals = implicit_fn(torch.stack(plane_intermediate).cuda()).cpu()
    timer.stop()
    print("CHECK VALS", check_vals.min().item(), check_vals.max().item())
    timer.start("E2")
    plane_normals = normal_fn(torch.stack(plane_intermediate).cuda()).cpu()
    timer.stop()
    print("DONE COMPUTING NORMALS")

    planes = []
    debug_points = []
    debug_normals = []
    debug_colors = []

    for i in tqdm(range(len(plane_intermediate))):
        timer.start("F1")
        plane_normal = plane_normals[i, :]
        plane_pos = plane_intermediate[i]
        debug_points.append(plane_pos.cpu().numpy())
        debug_normals.append(plane_normal.cpu().numpy())
        plane_val = check_vals[i].item()
        debug_colors.append(colormaps["PiYG"](plane_val + 0.5)[:3])
        planes.append(Plane(plane_normal, -torch.sum(plane_pos * plane_normal), plane_pos))
        timer.stop()
        # print(torch.cpu.max_memory_allocated())

    corner_points = []
    corner_colors = []
    for i in tqdm(range(corner_positions.shape[0])):
        timer.start("F2")
        z = corner_positions[i, -1].item()
        if z > -0.5 and z < 0.5:
            corner_points.append(corner_positions[i].cpu().numpy())
            corner_colors.append(colormaps["PiYG"](corner_values[i].item() + 0.5)[:3])
        timer.stop()

    vertices = []
    triangles = []
    for i in range(len(ids)):
        timer.start("G1")
        local_planes = [planes[j] for j in matched_planes[ids[i]]]
        simple_center = sum([planes[j].point for j in matched_planes[ids[i]]]) / len(matched_planes[ids[i]])
        # vertex = simple_center
        # simple_center = None
        vertex = fit_to_planes(local_planes, simple_center)
        # vertex = og.centers[ids[i]]
        vertices.append(vertex.cpu().numpy())
        timer.stop()

    dual_num = len(vertices)
    # vertices += debug_points

    for j in range(len(faces)):
        f = faces[j]
        # for i in range(len(f)):
        #     timer.start("H1")
        #     triangles.append((dual_num + j, used_ids[f[i]], used_ids[f[(i+1) % len(f)]]))
        #     timer.stop()
        for i in range(1, len(f) - 1):
            timer.start("H1")
            triangles.append((used_ids[f[0]], used_ids[f[i]], used_ids[f[i + 1]]))
            timer.stop()

    timer.dump()

    vertices = np.array(vertices)
    triangles = np.array(triangles)

    print("vertices", vertices.shape)

    print("triangles", triangles.shape)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(debug_points))
    pcd.normals = o3d.utility.Vector3dVector(np.array(debug_normals))
    pcd.colors = o3d.utility.Vector3dVector(np.array(debug_colors))
    # o3d.io.write_point_cloud("new_dual_contouring_debug.ply", pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(corner_points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(corner_colors))
    # o3d.io.write_point_cloud("new_dual_contouring_debug_corners.ply", pcd)
    return mesh


def sphere_function(xyz):
    if len(xyz.shape) == 1:
        xyz = xyz[None, :]
    return torch.sum(xyz**2, dim=-1) - 5**2


def sphere_normal(xyz):
    if len(xyz.shape) == 1:
        xyz = xyz[None, :]
    return xyz / torch.clamp(torch.linalg.norm(xyz, dim=-1)[..., None], min=1e-20)


def cube_function(xyz):
    if len(xyz.shape) == 1:
        xyz = xyz[None, :]
    return torch.max(torch.abs(xyz), dim=-1).values - 2.8


def cube_normal(xyz):
    if len(xyz.shape) == 1:
        xyz = xyz[None, :]
    indices = torch.argmax(torch.abs(xyz), dim=-1)
    normals = torch.zeros_like(xyz)
    normals[torch.arange(normals.shape[0]), indices] = 1
    return normals


def plane_function(xyz):
    if len(xyz.shape) == 1:
        xyz = xyz[None, :]
    return xyz[:, -1] - 0.1


def plane_normal(xyz):
    if len(xyz.shape) == 1:
        xyz = xyz[None, :]
    normal = torch.zeros_like(xyz)
    normal[:, -1] = 1
    return normal


if __name__ == "__main__":
    set_seed(0)
    aabb = torch.stack([torch.Tensor([-8, -8, -8]), torch.Tensor([8, 8, 8])])
    num_points = 10000
    points = torch.rand((num_points, 3)) * (aabb[1:, :] - aabb[:1, :]) + aabb[:1, :]
    # points[:,-1] = 0.1
    # points = None
    mesh = dual_contour(sphere_function, sphere_normal, points, aabb, 4, 6)
    # mesh = dual_contour(cube_function, cube_normal, points, aabb, 4, 6)
    o3d.io.write_triangle_mesh("new_dual_contouring_sphere_debug.ply", mesh)
