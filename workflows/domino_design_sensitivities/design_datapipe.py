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

"""
This is the datapipe to read OpenFoam files (vtp/vtu/stl) and save them as point clouds
in npy format.

The datapipe processes surface meshes to create structured representations suitable for
machine learning tasks, computing various geometric properties and signed distance fields.
"""

from typing import Literal, Sequence

import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from cuml.neighbors import NearestNeighbors
from torch.utils.data import Dataset

from physicsnemo.models.domino.utils import (
    calculate_center_of_mass,
    create_grid,
    normalize,
)
from physicsnemo.nn.functional import signed_distance_field
import torch


### PhysicsNeMo v2 made several DoMINO utilities torch-only. The rest of this
### datapipe is numpy-native (it feeds cuml's NearestNeighbors and uses
### ``np.array`` / ``rng.rand`` extensively), so we keep two small numpy-in /
### numpy-out wrappers below to isolate the torch hop at the boundary.


def _compute_sdf_np(
    mesh_vertices: np.ndarray,
    mesh_indices: np.ndarray,
    input_points: np.ndarray,
    use_sign_winding_number: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute signed distance + closest hit points from a triangle mesh.

    Thin numpy-in / numpy-out wrapper around the v2
    ``physicsnemo.nn.functional.signed_distance_field`` op, which is itself
    torch-only and always returns ``(sdf, hit_points)`` (no
    ``include_hit_points`` kwarg).

    Args:
        mesh_vertices: Mesh vertex coordinates, shape ``(n_vertices, 3)``.
        mesh_indices: Triangle connectivity as flattened indices, shape
            ``(3 * n_faces,)``; or face triplets, shape ``(n_faces, 3)``.
        input_points: Query points at which to evaluate the SDF, shape
            ``(..., 3)``.
        use_sign_winding_number: Use the winding-number sign convention
            (works on non-watertight meshes). Defaults to ``True``, matching
            the prior behavior of every call site in this datapipe.

    Returns:
        Tuple ``(sdf, hit_points)`` of numpy arrays. ``sdf`` has shape
        ``input_points.shape[:-1]`` and ``hit_points`` has shape
        ``input_points.shape``. Returned on CPU regardless of input device.
    """
    sdf, hit_points = signed_distance_field(
        mesh_vertices=torch.as_tensor(mesh_vertices),
        mesh_indices=torch.as_tensor(mesh_indices),
        input_points=torch.as_tensor(input_points),
        use_sign_winding_number=use_sign_winding_number,
    )
    return sdf.cpu().numpy(), hit_points.cpu().numpy()


def _create_grid_np(
    max_coords: np.ndarray,
    min_coords: np.ndarray,
    resolution: Sequence[int],
) -> np.ndarray:
    """Build a regular 3D grid via the v2 ``create_grid`` helper.

    The v2 ``physicsnemo.models.domino.utils.create_grid`` is torch-typed (it
    reads ``.dtype`` and ``.device`` off its first arg), so we wrap it for
    numpy callers. Output dtype is float32 to match the rest of the datapipe.

    Args:
        max_coords: Upper bounds ``[x_max, y_max, z_max]``, shape ``(3,)``.
        min_coords: Lower bounds ``[x_min, y_min, z_min]``, shape ``(3,)``.
        resolution: Number of grid points along each axis, length 3.

    Returns:
        Grid coordinates of shape ``(nx, ny, nz, 3)`` as ``np.float32``.
    """
    grid_t = create_grid(
        torch.as_tensor(np.asarray(max_coords), dtype=torch.float32),
        torch.as_tensor(np.asarray(min_coords), dtype=torch.float32),
        torch.as_tensor(np.asarray(resolution), dtype=torch.int32),
    )
    return grid_t.cpu().numpy()


class DesignDatapipe(Dataset):
    """PyTorch Dataset for processing surface meshes into DoMINO inputs."""

    def __init__(
        self,
        mesh: pv.PolyData,
        bounding_box: np.ndarray | tuple[NDArray[np.float32], NDArray[np.float32]],
        bounding_box_surface: (
            np.ndarray | tuple[NDArray[np.float32], NDArray[np.float32]]
        ),
        grid_resolution: Sequence[int],
        stencil_size: int = 7,
        seed: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize a DesignDatapipe dataset based on a surface mesh sample.

        Args:
            mesh: A PyVista PolyData mesh representing the surface geometry.
            bounding_box: A 2x3 numpy array containing the min and max coordinates of the volume
                bounding box. Shape: [[x_min, y_min, z_min], [x_max, y_max, z_max]].
            bounding_box_surface: A 2x3 numpy array containing the min and max coordinates of the
                surface bounding box. Shape: [[x_min, y_min, z_min], [x_max, y_max, z_max]].
            grid_resolution: A sequence of 3 integers specifying the number of points along each
                dimension (nx, ny, nz) for the structured grid.
            stencil_size: The size of the stencil used for local operations. Defaults to 7.
            seed: Random seed for reproducibility. Defaults to 0.

        Raises:
            ValueError: If grid_resolution does not contain exactly 3 values
            ValueError: If bounding_box or bounding_box_surface are not 2x3 arrays
        """
        if len(grid_resolution) != 3:
            raise ValueError("grid_resolution must contain exactly 3 values")

        self.mesh = mesh

        # Initialize random number generator, for reproducibility
        rng = np.random.RandomState(seed)

        # Initialize the output dictionary, which will store all data for the datapipe
        out_dict: dict[str, np.ndarray] = {}

        ### First, do computation that is required for all model_types
        length_scale = np.amax(self.mesh.points, 0) - np.amin(self.mesh.points, 0)
        stl_centers = self.mesh.cell_centers().points
        stl_faces = self.mesh.regular_faces
        mesh_indices_flattened = stl_faces.flatten()

        surface_areas = mesh.compute_cell_sizes(
            length=False, area=True, volume=False
        ).cell_data["Area"]

        surface_normals = -1.0 * np.array(mesh.cell_normals, dtype=np.float32)

        center_of_mass = (
            calculate_center_of_mass(
                torch.as_tensor(stl_centers), torch.as_tensor(surface_areas)
            )
            .detach()
            .cpu()
            .numpy()
        )

        s_max = np.asarray(bounding_box_surface[1])
        s_min = np.asarray(bounding_box_surface[0])

        v_max = np.asarray(bounding_box[1])
        v_min = np.asarray(bounding_box[0])

        nx, ny, nz = grid_resolution
        grid = _create_grid_np(v_max, v_min, grid_resolution)
        grid_reshaped = grid.reshape(nx * ny * nz, 3)

        sdf_grid, _ = _compute_sdf_np(
            mesh.points, mesh_indices_flattened, grid_reshaped
        )
        sdf_grid = sdf_grid.reshape(nx, ny, nz)

        surf_grid = _create_grid_np(s_max, s_min, grid_resolution)
        surf_grid_reshaped = surf_grid.reshape(nx * ny * nz, 3)

        sdf_surf_grid, _ = _compute_sdf_np(
            mesh.points, mesh_indices_flattened, surf_grid_reshaped
        )
        sdf_surf_grid = sdf_surf_grid.reshape(nx, ny, nz)

        # Sample surface_vertices
        grid = normalize(grid, v_max, v_min)
        surf_grid = normalize(surf_grid, s_max, s_min)

        surface_mesh_centers = stl_centers

        knn = NearestNeighbors(n_neighbors=stencil_size, algorithm="rbc")
        knn.fit(surface_mesh_centers)
        indices = knn.kneighbors(surface_mesh_centers, return_distance=False)

        ## CPU implementation of the above CUML neighbor-finding, as a backup
        # from scipy.spatial import KDTree
        # interp_func = KDTree(surface_mesh_centers)
        # distances, indices = interp_func.query(surface_mesh_centers, k=stencil_size)

        surface_mesh_neighbors = surface_mesh_centers[indices]
        surface_mesh_neighbors = surface_mesh_neighbors[:, 1:] + 1e-6
        surface_neighbors_normals = surface_normals[indices]
        surface_neighbors_normals = surface_neighbors_normals[:, 1:]
        surface_neighbors_areas = surface_areas[indices]
        surface_neighbors_areas = surface_neighbors_areas[:, 1:]

        pos_normals_com_surface = surface_mesh_centers - center_of_mass

        surface_mesh_centers = normalize(surface_mesh_centers, s_max, s_min)
        surface_mesh_neighbors = normalize(surface_mesh_neighbors, s_max, s_min)

        # Volume processing
        volume_coordinates = (v_max - v_min) * rng.rand(1000, 3) + v_min

        sdf_nodes, sdf_node_closest_point = _compute_sdf_np(
            mesh.points, mesh_indices_flattened, volume_coordinates
        )
        sdf_nodes = sdf_nodes.reshape(-1, 1)
        pos_normals_closest = volume_coordinates - sdf_node_closest_point
        pos_volume_center_of_mass = volume_coordinates - center_of_mass
        volume_coordinates = normalize(volume_coordinates, v_max, v_min)
        vol_grid_max_min = np.float32(np.asarray([v_min, v_max]))
        surf_grid_max_min = np.float32(np.asarray([s_min, s_max]))

        self.out_dict = dict(
            pos_volume_closest=pos_normals_closest,
            pos_volume_center_of_mass=pos_volume_center_of_mass,
            pos_surface_center_of_mass=pos_normals_com_surface,
            geometry_coordinates=stl_centers,
            grid=grid,
            surf_grid=surf_grid,
            sdf_grid=sdf_grid,
            sdf_surf_grid=sdf_surf_grid,
            sdf_nodes=sdf_nodes,
            surface_mesh_centers=surface_mesh_centers,
            surface_mesh_neighbors=surface_mesh_neighbors,
            surface_normals=surface_normals,
            surface_areas=surface_areas,
            surface_neighbors_normals=surface_neighbors_normals,
            surface_neighbors_areas=surface_neighbors_areas,
            volume_mesh_centers=volume_coordinates,
            volume_min_max=vol_grid_max_min,
            surface_min_max=surf_grid_max_min,
            length_scale=length_scale,
        )

        self.out_dict = {
            k: torch.from_numpy(v).type(torch.float32).to(device)
            for k, v in self.out_dict.items()
        }

    def __len__(self) -> int:
        """Return the number of faces in the mesh."""
        return self.mesh.n_faces_strict

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing surface mesh data for the specified index
        """
        keys: list[str] = [
            "surface_mesh_centers",
            "surface_mesh_neighbors",
            "surface_normals",
            "surface_neighbors_normals",
            "surface_areas",
            "surface_neighbors_areas",
            "pos_surface_center_of_mass",
        ]

        return {k: self.out_dict[k][idx] for k in keys}


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    mesh: pv.PolyData = pv.read("./geometries/drivaer_1_single_solid_decimated3.stl")
    bounding_box: np.ndarray = np.array([[-3.5, -2.25, -0.32], [8.5, 2.25, 3.00]])
    bounding_box_surface: np.ndarray = np.array([[-1.1, -1.2, -0.32], [4.5, 1.2, 1.2]])

    fd = DesignDatapipe(
        mesh=mesh,
        bounding_box=bounding_box,
        bounding_box_surface=bounding_box_surface,
        grid_resolution=[128, 64, 48],
    )

    train_dataloader = DataLoader(fd, batch_size=256_000, shuffle=False)

    for i, sample_batched in enumerate(train_dataloader):
        print(f"{i=}, {sample_batched['surface_mesh_centers'].shape=}")
