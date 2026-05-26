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
Preprocesses a PyVista surface mesh into the input-tensor dict consumed by
:class:`physicsnemo.models.domino.model.DoMINO`.

Given an STL-derived ``pv.PolyData`` plus volume and surface bounding
boxes, this datapipe computes signed-distance fields on a structured
grid, cell centroids and area-weighted neighborhood stencils for the
surface mesh, and (optionally) a sparse random sampling of volume query
points. All outputs are emitted as ``torch.float32`` tensors keyed by
the names the DoMINO model's ``forward()`` expects.
"""

from typing import Sequence

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
    device: torch.device = torch.device("cpu"),
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
        device: Torch device used for the SDF computation. The PhysicsNeMo
            v2 ``signed_distance_field`` op dispatches its Warp kernel to
            the device of its inputs, so leaving this at the default
            ``cpu`` would silently run an O(n_query * n_faces) BVH query
            on a single CPU core. Pass a CUDA device on GPU machines.

    Returns:
        Tuple ``(sdf, hit_points)`` of numpy arrays. ``sdf`` has shape
        ``input_points.shape[:-1]`` and ``hit_points`` has shape
        ``input_points.shape``. Returned on CPU regardless of input device.
    """
    sdf, hit_points = signed_distance_field(
        mesh_vertices=torch.as_tensor(mesh_vertices, device=device),
        mesh_indices=torch.as_tensor(mesh_indices, device=device),
        input_points=torch.as_tensor(input_points, device=device),
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

    ### The set of surface tensors that vary across the inner DataLoader
    ### batches. Hoisted here so :mod:`main` and :meth:`__getitem__`
    ### stay in lockstep without duplicating the list.
    SURFACE_KEYS: tuple[str, ...] = (
        "surface_mesh_centers",
        "surface_mesh_neighbors",
        "surface_normals",
        "surface_neighbors_normals",
        "surface_areas",
        "surface_neighbors_areas",
        "pos_surface_center_of_mass",
    )

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
        produce_volume_inputs: bool = True,
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
            device: Device on which to place the emitted tensors.
            produce_volume_inputs: Whether to also emit the volume-side tensors
                (``sdf_nodes``, ``volume_mesh_centers``, ``pos_volume_closest``,
                ``pos_volume_center_of_mass``). Set to ``False`` for surface-only
                checkpoints to avoid sampling random volume points the model
                does not read. Defaults to ``True``.

        Raises:
            ValueError: If grid_resolution does not contain exactly 3 values.
        """
        if len(grid_resolution) != 3:
            raise ValueError("grid_resolution must contain exactly 3 values")

        self.mesh = mesh

        # Initialize random number generator, for reproducibility
        rng = np.random.RandomState(seed)

        ### First, do computation that is required for all model_types
        length_scale = np.amax(self.mesh.points, 0) - np.amin(self.mesh.points, 0)
        stl_centers = self.mesh.cell_centers().points
        stl_faces = self.mesh.regular_faces
        mesh_indices_flattened = stl_faces.flatten()

        surface_areas = mesh.compute_cell_sizes(
            length=False, area=True, volume=False
        ).cell_data["Area"]

        ### DoMINO was trained with inward-pointing surface normals (the
        ### opposite of PyVista's outward-pointing convention), so flip
        ### the sign here to match the training preprocessing.
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
            mesh.points, mesh_indices_flattened, grid_reshaped, device=device
        )
        sdf_grid = sdf_grid.reshape(nx, ny, nz)

        surf_grid = _create_grid_np(s_max, s_min, grid_resolution)
        surf_grid_reshaped = surf_grid.reshape(nx * ny * nz, 3)

        sdf_surf_grid, _ = _compute_sdf_np(
            mesh.points, mesh_indices_flattened, surf_grid_reshaped, device=device
        )
        sdf_surf_grid = sdf_surf_grid.reshape(nx, ny, nz)

        # Sample surface_vertices
        grid = normalize(grid, v_max, v_min)
        surf_grid = normalize(surf_grid, s_max, s_min)

        surface_mesh_centers = stl_centers

        knn = NearestNeighbors(n_neighbors=stencil_size, algorithm="rbc")
        knn.fit(surface_mesh_centers)
        indices = knn.kneighbors(surface_mesh_centers, return_distance=False)

        ### k-NN returns each query as its own nearest neighbor at column 0;
        ### slice it off so only true neighbors remain. The 1e-6 offset on
        ### `surface_mesh_neighbors` guards against exact coincidences that
        ### would later produce NaNs in geometry-encoding distance ratios.
        surface_mesh_neighbors = surface_mesh_centers[indices][:, 1:] + 1e-6
        surface_neighbors_normals = surface_normals[indices][:, 1:]
        surface_neighbors_areas = surface_areas[indices][:, 1:]

        pos_normals_com_surface = surface_mesh_centers - center_of_mass

        surface_mesh_centers = normalize(surface_mesh_centers, s_max, s_min)
        surface_mesh_neighbors = normalize(surface_mesh_neighbors, s_max, s_min)

        vol_grid_max_min = np.asarray([v_min, v_max], dtype=np.float32)
        surf_grid_max_min = np.asarray([s_min, s_max], dtype=np.float32)

        self.out_dict = dict(
            pos_surface_center_of_mass=pos_normals_com_surface,
            geometry_coordinates=stl_centers,
            grid=grid,
            surf_grid=surf_grid,
            sdf_grid=sdf_grid,
            sdf_surf_grid=sdf_surf_grid,
            surface_mesh_centers=surface_mesh_centers,
            surface_mesh_neighbors=surface_mesh_neighbors,
            surface_normals=surface_normals,
            surface_areas=surface_areas,
            surface_neighbors_normals=surface_neighbors_normals,
            surface_neighbors_areas=surface_neighbors_areas,
            volume_min_max=vol_grid_max_min,
            surface_min_max=surf_grid_max_min,
            length_scale=length_scale,
        )

        if produce_volume_inputs:
            ### Sample a sparse cloud of query points inside the volume
            ### bounding box, then compute their SDF, closest-mesh-point
            ### offset, and offset-from-center-of-mass. DoMINO only reads
            ### these when its `output_features_vol` head is enabled; for
            ### surface-only checkpoints we skip the work entirely.
            volume_coordinates = (v_max - v_min) * rng.rand(1000, 3) + v_min
            sdf_nodes, sdf_node_closest_point = _compute_sdf_np(
                mesh.points, mesh_indices_flattened, volume_coordinates, device=device
            )
            sdf_nodes = sdf_nodes.reshape(-1, 1)
            pos_normals_closest = volume_coordinates - sdf_node_closest_point
            pos_volume_center_of_mass = volume_coordinates - center_of_mass
            volume_coordinates = normalize(volume_coordinates, v_max, v_min)
            self.out_dict.update(
                sdf_nodes=sdf_nodes,
                pos_volume_closest=pos_normals_closest,
                pos_volume_center_of_mass=pos_volume_center_of_mass,
                volume_mesh_centers=volume_coordinates,
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
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing the surface tensors listed in
            :attr:`SURFACE_KEYS`, indexed by ``idx``.
        """
        return {k: self.out_dict[k][idx] for k in self.SURFACE_KEYS}
