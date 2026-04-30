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

"""kNN-based interpolation from prediction points to full mesh."""

from typing import Final, Union

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

# Targets that coincide with (or duplicate) a neighbor use that neighbor exactly; tolerant to FP noise.
_COINCIDENT_ATOL: Final[float] = 1e-12


def _combine_field_neighbors_idw(
    distances: np.ndarray,
    field_neighbors: np.ndarray,
    *,
    epsilon: float = 1e-8,
    coincident_atol: float = _COINCIDENT_ATOL,
) -> np.ndarray:
    """Weighted kNN interpolation from stacked neighbor field values.

    If any neighbor distance is at or below ``coincident_atol``, that target row uses the
    **first** such neighbor's value exactly (columns are nearest-first from kNN). Otherwise
    inverse-distance weights ``1/(d+epsilon)`` are normalized per row.
    """
    coincident = distances <= coincident_atol
    has = coincident.any(axis=1)
    first_j = np.argmax(coincident, axis=1)
    m_ix = np.where(has)[0]

    weights = 1.0 / (distances + epsilon)
    norm_w = weights / np.sum(weights, axis=1, keepdims=True)
    if field_neighbors.ndim == 2:
        out = np.sum(norm_w * field_neighbors, axis=1)
        if m_ix.size:
            out[m_ix] = field_neighbors[m_ix, first_j[m_ix]]
        return out
    out = np.sum(norm_w[:, :, np.newaxis] * field_neighbors, axis=1)
    if m_ix.size:
        out[m_ix] = field_neighbors[m_ix, first_j[m_ix]]
    return out


def interpolate_to_mesh(
    mesh_points: np.ndarray,
    pred_coords: np.ndarray,
    pressure_pred: Union[torch.Tensor, np.ndarray],
    wss_pred: Union[torch.Tensor, np.ndarray],
    k: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate pressure and WSS from prediction points to full mesh via weighted kNN.

    When ``k>1``, neighbors are inverse-distance weighted. If a target lies within
    ``1e-12`` of a source point (typical duplicate / exact hit), that neighbor's value
    is used exactly instead of relying on ``epsilon`` in ``1/(d+epsilon)``.

    Args:
        mesh_points: Target coordinates [M, 3].
        pred_coords: Prediction point coordinates [N, 3].
        pressure_pred: Pressure at prediction points [N] or [N, 1].
        wss_pred: Shear stress at prediction points [N, 3].
        k: Number of nearest neighbors for weighting.

    Returns:
        pressure_mesh: [M] or [M, 1] float32, pressure interpolated to mesh.
        wss_mesh: [M, 3] float32, shear stress interpolated to mesh.
    """
    if isinstance(pressure_pred, torch.Tensor):
        pressure_pred = pressure_pred.cpu().numpy()
    if isinstance(wss_pred, torch.Tensor):
        wss_pred = wss_pred.cpu().numpy()
    pressure_pred = np.asarray(pressure_pred, dtype=np.float32)
    wss_pred = np.asarray(wss_pred, dtype=np.float32)
    if pressure_pred.ndim == 2:
        pressure_pred = pressure_pred.squeeze(-1)
    pred_coords = np.asarray(pred_coords, dtype=np.float64)
    mesh_points = np.asarray(mesh_points, dtype=np.float64)

    nbrs = NearestNeighbors(n_neighbors=min(k, len(pred_coords)), algorithm="ball_tree").fit(
        pred_coords
    )
    distances, indices = nbrs.kneighbors(mesh_points)

    if k == 1:
        indices = indices.flatten()
        p_mesh = pressure_pred[indices]
        wss_mesh = wss_pred[indices]
    else:
        p_mesh = _combine_field_neighbors_idw(distances, pressure_pred[indices])
        wss_mesh = _combine_field_neighbors_idw(distances, wss_pred[indices])
    return p_mesh.astype(np.float32), wss_mesh.astype(np.float32)
