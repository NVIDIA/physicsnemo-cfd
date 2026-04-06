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

from typing import Union

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def interpolate_to_mesh(
    mesh_points: np.ndarray,
    pred_coords: np.ndarray,
    pressure_pred: Union[torch.Tensor, np.ndarray],
    wss_pred: Union[torch.Tensor, np.ndarray],
    k: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate pressure and WSS from prediction points to full mesh via weighted kNN.

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
        epsilon = 1e-8
        weights = 1.0 / (distances + epsilon)
        norm_weights = weights / np.sum(weights, axis=1, keepdims=True)
        p_mesh = np.sum(norm_weights * pressure_pred[indices], axis=1)
        wss_mesh = np.sum(
            norm_weights[:, :, np.newaxis] * wss_pred[indices], axis=1
        )
    return p_mesh.astype(np.float32), wss_mesh.astype(np.float32)
