# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree


def _create_nbrs_surface(coords_source, n_neighbors=5, device="cpu"):
    if device == "cpu":
        nbrs_surface = cKDTree(coords_source)
    elif device == "gpu":
        import cupy as cp
        from cuml.neighbors import NearestNeighbors as NearestNeighborsGPU

        if not isinstance(coords_source, cp.ndarray):
            coords_source = cp.asarray(coords_source)

        nbrs_surface = NearestNeighborsGPU(
            n_neighbors=n_neighbors, algorithm="auto"
        ).fit(coords_source)
    return nbrs_surface


def _interpolate(
    nbrs_surface,
    coords_target,
    field,
    device="cpu",
    batch_size=1_000_000,
    n_neighbors=5,
):

    if device == "cpu":
        distances, indices = nbrs_surface.query(coords_target, k=n_neighbors, workers=8)

        epsilon = 1e-8
        weights = 1 / (distances + epsilon)
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        normalized_weights = weights / weights_sum

        field_neighbors = field[indices]
        if len(field.shape) == 1:
            field_interp = np.sum(normalized_weights * field_neighbors, axis=1)
        else:
            field_interp = np.sum(
                normalized_weights[:, :, np.newaxis] * field_neighbors, axis=1
            )
    elif device == "gpu":
        import cupy as cp

        if not isinstance(field, cp.ndarray):
            field = cp.asarray(field)

        if len(field.shape) == 1:
            field_interp = np.zeros((coords_target.shape[0],))
        else:
            field_interp = np.zeros((coords_target.shape[0], field.shape[1]))

        for i in range(0, coords_target.shape[0], batch_size):
            batch_pts = cp.asarray(coords_target[i : i + batch_size])
            distances, indices = nbrs_surface.kneighbors(batch_pts)
            epsilon = 1e-8
            weights = 1 / (distances + epsilon)
            weights_sum = cp.sum(weights, axis=1, keepdims=True)
            normalized_weights = weights / weights_sum
            field_neighbors = field[indices]
            if len(field.shape) == 1:
                field_interp[i : i + batch_size] = cp.asnumpy(
                    cp.sum(normalized_weights * field_neighbors, axis=1)
                )
            else:
                field_interp[i : i + batch_size] = cp.asnumpy(
                    cp.sum(
                        normalized_weights[:, :, cp.newaxis] * field_neighbors,
                        axis=1,
                    )
                )

    return field_interp


def interpolate_mesh_to_pc(pc, mesh, fields_to_interpolate, mesh_dtype="cell"):
    """Interpolate mesh results on a point cloud using inverse weighted kNN

    Parameters
    ----------
    pc :
        Point Cloud to interpolate values on (PyVista Dataset)
    mesh :
        Mesh for the source values (PyVista Dataset)
    fields_to_interpolate :
        List of fields (str) to interpolate (must be present in the mesh dataset)
    mesh_dtype :
        Whether the mesh fields are of point or cell type. Default cell.

    Returns
    -------
    _type_
        Point cloud with interpolated values
    """
    k = 5

    if mesh_dtype == "point":
        source_points = mesh.points
    elif mesh_dtype == "cell":
        cell_centers = mesh.cell_centers()
        source_points = cell_centers.points

    nbrs_surface = _create_nbrs_surface(source_points, n_neighbors=k)

    if mesh_dtype == "point":
        # TODO: Possible to vectorize
        for field in fields_to_interpolate:
            pc.point_data[field] = _interpolate(
                nbrs_surface, pc.points, mesh.point_data[field], n_neighbors=k
            )
    elif mesh_dtype == "cell":
        # TODO: Possible to vectorize
        for field in fields_to_interpolate:
            pc.point_data[field] = _interpolate(
                nbrs_surface, pc.points, mesh.cell_data[field], n_neighbors=k
            )

    return pc
