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

import warnings

import numpy as np
import pyvista as pv
import torch
from scipy.spatial import cKDTree

from physicsnemo.nn.functional.neighbors.knn import knn


# ---------------------------------------------------------------------------
# Deprecated legacy helpers — kept for backward compatibility with
# workflows/deprecated/bench_example notebooks. Will be removed when that workflow is retired.
# ---------------------------------------------------------------------------

def _create_nbrs_surface(coords_source, n_neighbors=5, device="cpu"):
    """Deprecated: use ``interpolate_mesh_to_pc`` or ``physicsnemo.nn.functional.neighbors.knn``."""
    warnings.warn(
        "_create_nbrs_surface is deprecated; use interpolate_mesh_to_pc or physicsnemo knn.",
        DeprecationWarning,
        stacklevel=2,
    )
    if device == "cpu":
        return cKDTree(coords_source)
    elif device == "gpu":
        import cupy as cp
        from cuml.neighbors import NearestNeighbors as NearestNeighborsGPU

        if not isinstance(coords_source, cp.ndarray):
            coords_source = cp.asarray(coords_source)
        return NearestNeighborsGPU(n_neighbors=n_neighbors, algorithm="auto").fit(coords_source)
    return cKDTree(coords_source)


def _interpolate(nbrs_surface, coords_target, field, device="cpu", batch_size=1_000_000, n_neighbors=5):
    """Deprecated: use ``interpolate_mesh_to_pc`` or ``physicsnemo.nn.functional.neighbors.knn``."""
    warnings.warn(
        "_interpolate is deprecated; use interpolate_mesh_to_pc or physicsnemo knn.",
        DeprecationWarning,
        stacklevel=2,
    )
    if device == "cpu":
        distances, indices = nbrs_surface.query(coords_target, k=n_neighbors, workers=8)
        epsilon = 1e-8
        weights = 1 / (distances + epsilon)
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        normalized_weights = weights / weights_sum
        field_neighbors = field[indices]
        if len(field.shape) == 1:
            return np.sum(normalized_weights * field_neighbors, axis=1)
        return np.sum(normalized_weights[:, :, np.newaxis] * field_neighbors, axis=1)
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
                    cp.sum(normalized_weights[:, :, cp.newaxis] * field_neighbors, axis=1)
                )
        return field_interp


# ---------------------------------------------------------------------------
# Current implementation using physicsnemo kNN
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> torch.device:
    """Map legacy ``"gpu"`` flag and torch device strings to a ``torch.device``."""
    if device == "gpu":
        return torch.device("cuda")
    return torch.device(device)


def _idw_interpolate(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    field: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Inverse-distance-weighted interpolation using PhysicsNeMo kNN.

    Parameters
    ----------
    source_points : torch.Tensor
        Source coordinates (N, 3).
    target_points : torch.Tensor
        Target coordinates (M, 3).
    field : np.ndarray
        Field values at source points — (N,) for scalars or (N, C) for vectors.
    k : int
        Number of nearest neighbors.

    Returns
    -------
    np.ndarray
        Interpolated field values at target points.
    """
    indices, distances = knn(source_points, target_points, k)

    indices_np = indices.cpu().numpy()
    distances_np = distances.cpu().numpy().astype(np.float64)

    epsilon = 1e-8
    weights = 1.0 / (distances_np + epsilon)
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    normalized_weights = weights / weights_sum

    field_neighbors = field[indices_np]
    if field.ndim == 1:
        return np.sum(normalized_weights * field_neighbors, axis=1)
    return np.sum(normalized_weights[:, :, np.newaxis] * field_neighbors, axis=1)


def interpolate_mesh_to_pc(pc, mesh, fields_to_interpolate, mesh_dtype="cell", device="cpu"):
    """Interpolate mesh results on a point cloud using inverse weighted kNN.

    Uses ``physicsnemo.nn.functional.neighbors.knn`` which auto-dispatches to
    cuML on CUDA and scipy on CPU.

    Parameters
    ----------
    pc :
        Point Cloud to interpolate values on (PyVista Dataset).
    mesh :
        Mesh for the source values (PyVista Dataset).
    fields_to_interpolate :
        List of fields (str) to interpolate (must be present in the mesh dataset).
    mesh_dtype :
        Whether the mesh fields are of point or cell type. Default ``"cell"``.
    device :
        ``"cpu"``, ``"gpu"``, or a torch device string (e.g. ``"cuda:0"``).
        ``"gpu"`` is mapped to ``"cuda"`` for backward compatibility.

    Returns
    -------
    pv.DataSet
        Point cloud with interpolated values.
    """
    k = 5
    dev = _resolve_device(device)

    if mesh_dtype == "point":
        source_points = np.asarray(mesh.points, dtype=np.float32)
    elif mesh_dtype == "cell":
        source_points = np.asarray(mesh.cell_centers().points, dtype=np.float32)

    source_t = torch.tensor(source_points, dtype=torch.float32, device=dev)
    target_t = torch.tensor(np.asarray(pc.points, dtype=np.float32), dtype=torch.float32, device=dev)

    data = mesh.point_data if mesh_dtype == "point" else mesh.cell_data
    for field_name in fields_to_interpolate:
        field_arr = np.asarray(data[field_name])
        pc.point_data[field_name] = _idw_interpolate(source_t, target_t, field_arr, k=k)

    return pc


def interpolate_point_data_to_cell_centers(
    mesh: pv.DataSet,
    field_names: list[str],
    *,
    k: int = 5,
    device: str = "cpu",
) -> pv.DataSet:
    """Interpolate fields defined on mesh vertices to cell centers via kNN inverse-distance weighting.

    Uses the same ``knn`` + IDW path as :func:`interpolate_mesh_to_pc` (source = ``mesh.points``,
    targets = ``mesh.cell_centers().points``). Each interpolated array is written to ``mesh.cell_data``
    and removed from ``mesh.point_data`` so downstream metrics can use ``dtype=\"cell\"`` (e.g.
    :func:`~physicsnemo.cfd.postprocessing_tools.metrics.aero_forces.compute_drag_and_lift`).

    Parameters
    ----------
    mesh :
        Surface (or any dataset with points and cells). Fields must live in ``mesh.point_data``.
    field_names :
        VTK array names to promote. Names missing from ``point_data`` are skipped.
    k :
        Number of neighbors for IDW (default 5).
    device :
        ``\"cpu\"``, ``\"gpu\"`` (maps to CUDA), or a torch device string.

    Returns
    -------
    pv.DataSet
        The same ``mesh`` instance, modified in place.
    """
    dev = _resolve_device(device)
    source_points = np.asarray(mesh.points, dtype=np.float32)
    target_points = np.asarray(mesh.cell_centers().points, dtype=np.float32)
    source_t = torch.tensor(source_points, dtype=torch.float32, device=dev)
    target_t = torch.tensor(target_points, dtype=torch.float32, device=dev)
    n_pt = int(mesh.n_points)

    for field_name in field_names:
        if field_name not in mesh.point_data:
            continue
        field_arr = np.asarray(mesh.point_data[field_name])
        if field_arr.ndim == 1:
            if field_arr.shape[0] != n_pt:
                raise ValueError(
                    f"Field {field_name!r}: expected {n_pt} point values, got shape {field_arr.shape}"
                )
        elif field_arr.ndim == 2:
            if field_arr.shape[0] != n_pt:
                raise ValueError(
                    f"Field {field_name!r}: expected ({n_pt}, C) at points, got {field_arr.shape}"
                )
        else:
            raise ValueError(f"Unsupported point array shape for {field_name!r}: {field_arr.shape}")

        mesh.cell_data[field_name] = _idw_interpolate(source_t, target_t, field_arr, k=k)
        try:
            mesh.point_data.remove(field_name)
        except AttributeError:
            del mesh.point_data[field_name]

    return mesh
