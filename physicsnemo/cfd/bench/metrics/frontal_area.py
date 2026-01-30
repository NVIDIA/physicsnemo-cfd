# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import warp as wp
import numpy as np
import pyvista as pv
from shapely.geometry import Polygon
from shapely.ops import unary_union


@wp.kernel
def _rasterize_triangles(
    vertices: wp.array(dtype=wp.vec3),
    faces: wp.array(dtype=wp.vec3i),
    normal: wp.vec3,
    indices: wp.vec2i,
    grid_origin: wp.vec2,
    grid_resolution: int,
    cell_size: float,
    grid: wp.array(dtype=int),
):
    """Warp kernel to rasterize triangles"""
    tid = wp.tid()

    i, j, k = faces[tid][0], faces[tid][1], faces[tid][2]
    A, B, C = vertices[i], vertices[j], vertices[k]

    # project vertices onto the plane
    A_proj = wp.vec2(A[indices[0]], A[indices[1]])
    B_proj = wp.vec2(B[indices[0]], B[indices[1]])
    C_proj = wp.vec2(C[indices[0]], C[indices[1]])

    # calculate bounding box of the projected triangle
    min_x = wp.min(A_proj[0], wp.min(B_proj[0], C_proj[0]))
    max_x = wp.max(A_proj[0], wp.max(B_proj[0], C_proj[0]))
    min_y = wp.min(A_proj[1], wp.min(B_proj[1], C_proj[1]))
    max_y = wp.max(A_proj[1], wp.max(B_proj[1], C_proj[1]))

    min_cell_x = int((min_x - grid_origin[0]) / cell_size)
    max_cell_x = int((max_x - grid_origin[0]) / cell_size)
    min_cell_y = int((min_y - grid_origin[1]) / cell_size)
    max_cell_y = int((max_y - grid_origin[1]) / cell_size)

    # clamp to grid boundaries
    min_cell_x = wp.clamp(min_cell_x, 0, grid_resolution - 1)
    max_cell_x = wp.clamp(max_cell_x, 0, grid_resolution - 1)
    min_cell_y = wp.clamp(min_cell_y, 0, grid_resolution - 1)
    max_cell_y = wp.clamp(max_cell_y, 0, grid_resolution - 1)

    # mark grid cells within the bounding box
    for x in range(min_cell_x, max_cell_x + 1):
        for y in range(min_cell_y, max_cell_y + 1):
            wp.atomic_add(grid, y * grid_resolution + x, 1)


def _compute_frontal_area_rasterization(
    mesh_vertices, mesh_faces, direction="x", grid_resolution=512
):
    """compute approx frontal area via rasterization"""
    direction_map = {
        "x": ((1, 2), wp.vec3(1.0, 0.0, 0.0)),
        "y": ((0, 2), wp.vec3(0.0, 1.0, 0.0)),
        "z": ((0, 1), wp.vec3(0.0, 0.0, 1.0)),
    }

    if direction not in direction_map:
        raise ValueError("Direction must be x, y, or z only")

    indices, normal = direction_map[direction]
    indices = wp.vec2i(indices[0], indices[1])

    vertices = wp.array(mesh_vertices, dtype=wp.vec3)
    faces = wp.array(mesh_faces, dtype=wp.vec3i)

    grid_origin = np.min(mesh_vertices[:, indices], axis=0)
    grid_extent = np.ptp(mesh_vertices[:, indices], axis=0)
    cell_size = max(grid_extent) / grid_resolution
    grid = wp.zeros(grid_resolution * grid_resolution, dtype=int)

    # Launch kernel
    wp.launch(
        kernel=_rasterize_triangles,
        dim=len(mesh_faces),
        inputs=[
            vertices,
            faces,
            normal,
            indices,
            grid_origin,
            grid_resolution,
            cell_size,
            grid,
        ],
    )

    # Sum up covered grid cells
    grid_np = grid.numpy()
    covered_cells = np.sum(grid_np > 0)
    area = covered_cells * (cell_size**2)

    return area


def compute_frontal_area(mesh, direction="x", method="exact", grid_resolution=512):
    """Compute frontal area of a mesh (stl/vtp)

    Parameters
    ----------
    mesh :
        PyVista Dataset
    direction : str, optional
        Direction for projection (x, y or z), by default "x"
    method : str, optional
        Wether to use exact or approximate method. Approximate method uses rasterization, by default "exact"
    grid_resolution : int, optional
        Grid resolution for rasterization. Only used if the method is approximate, by default 512

    Returns
    -------
    _type_
        Returns the frontal area of the mesh

    """
    direction_map = {
        "x": (1, 2),
        "y": (0, 2),
        "z": (0, 1),
    }

    if direction not in direction_map:
        raise ValueError("Direction must be x, y, or z only")

    indices = direction_map[direction]

    if method == "exact":
        mesh = mesh.triangulate()
        mesh_vertices = mesh.points
        mesh_faces = mesh.faces.reshape(-1, 4)[:, 1:]

        polygons = []
        for idx, face in enumerate(mesh_faces):
            verts = mesh_vertices[face]
            polygon = Polygon(verts[:, indices])  # Use the projected 2D indices
            polygons.append(polygon)

        merged_polygon = unary_union(polygons)
        area = merged_polygon.area
    elif method == "approximate":
        mesh = mesh.triangulate()
        mesh_vertices = mesh.points
        mesh_faces = mesh.faces.reshape(-1, 4)[:, 1:]

        area = _compute_frontal_area_rasterization(
            mesh_vertices,
            mesh_faces,
            direction=direction,
            grid_resolution=grid_resolution,
        )
    else:
        raise ValueError("Method must be either exact or approximate")

    return area


if __name__ == "__main__":
    # Example Usage
    # sphere = pv.Sphere(radius=1.0, theta_resolution=1000, phi_resolution=1000)
    sphere = pv.read("../../test_scripts/drivaer_100_single_solid.stl")
    import time

    start_time = time.time()
    frontal_area = compute_frontal_area(sphere, direction="x", method="exact")
    print(
        f"Frontal Area of the Sphere (Exact) (Projection along x-axis): {frontal_area}, time taken: {time.time() - start_time}"
    )

    start_time = time.time()
    frontal_area = compute_frontal_area(
        sphere, direction="x", method="approximate", grid_resolution=1024
    )
    print(
        f"Frontal Area of the Sphere (Approx.) (Projection along x-axis): {frontal_area}, time taken: {time.time() - start_time}"
    )
