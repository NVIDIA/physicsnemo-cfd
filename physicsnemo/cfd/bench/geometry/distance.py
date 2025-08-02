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

import numpy as np
from scipy.spatial import cKDTree
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
import cupy as cp


def hausdorff_distance(point_cloud_1, point_cloud_2, method="kdtree"):
    """
    Compute the Hausdorff distance between two large point clouds efficiently.

    Parameters:
    point_cloud_1: numpy array of shape (n_points, n_dimensions)
    point_cloud_2: numpy array of shape (m_points, n_dimensions)
    method: 'kdtree' or 'cuml_knn' for nearest neighbor search

    Returns:
    hausdorff_dist: float, the Hausdorff distance
    """
    if method == "cuml_knn":
        return _hausdorff_cuml(point_cloud_1, point_cloud_2)
    elif method == "kdtree":
        return _hausdorff_kdtree(point_cloud_1, point_cloud_2)
    else:
        raise ValueError("Method must be 'cuml_knn' or 'kdtree'")


def _hausdorff_cuml(point_cloud_1, point_cloud_2):
    """Use cuML's NearestNeighbors for GPU-accelerated nearest neighbor search."""
    gpu_cloud_1 = cp.asarray(point_cloud_1, dtype=cp.float32)
    gpu_cloud_2 = cp.asarray(point_cloud_2, dtype=cp.float32)

    nn_2 = cuNearestNeighbors(n_neighbors=1, algorithm="auto").fit(gpu_cloud_2)
    min_distances_1_to_2, _ = nn_2.kneighbors(gpu_cloud_1)
    min_distances_1_to_2 = cp.asnumpy(min_distances_1_to_2).flatten()

    nn_1 = cuNearestNeighbors(n_neighbors=1, algorithm="auto").fit(gpu_cloud_1)
    min_distances_2_to_1, _ = nn_1.kneighbors(gpu_cloud_2)
    min_distances_2_to_1 = cp.asnumpy(min_distances_2_to_1).flatten()

    hausdorff_dist = max(np.max(min_distances_1_to_2), np.max(min_distances_2_to_1))

    return hausdorff_dist


def _hausdorff_kdtree(point_cloud_1, point_cloud_2):
    """Use scipy's cKDTree for efficient nearest neighbor search."""
    tree_2 = cKDTree(point_cloud_2)

    min_distances_1_to_2, _ = tree_2.query(point_cloud_1, k=1)

    tree_1 = cKDTree(point_cloud_1)

    min_distances_2_to_1, _ = tree_1.query(point_cloud_2, k=1)

    hausdorff_dist = max(np.max(min_distances_1_to_2), np.max(min_distances_2_to_1))

    return hausdorff_dist


def chamfer_distance(point_cloud_1, point_cloud_2, method="kdtree"):
    """
    Compute the Chamfer distance between two large point clouds efficiently.

    Parameters:
    point_cloud_1: numpy array of shape (n_points, n_dimensions)
    point_cloud_2: numpy array of shape (m_points, n_dimensions)
    method: 'kdtree' or 'cuml_knn' for nearest neighbor search

    Returns:
    chamfer_dist: float, the Chamfer distance
    """
    if method == "cuml_knn":
        return _chamfer_cuml(point_cloud_1, point_cloud_2)
    elif method == "kdtree":
        return _chamfer_kdtree(point_cloud_1, point_cloud_2)
    else:
        raise ValueError("Method must be 'cuml_knn' or 'kdtree'")


def _chamfer_cuml(point_cloud_1, point_cloud_2):
    """Use cuML's NearestNeighbors for GPU-accelerated Chamfer distance calculation."""
    gpu_cloud_1 = cp.asarray(point_cloud_1, dtype=cp.float32)
    gpu_cloud_2 = cp.asarray(point_cloud_2, dtype=cp.float32)

    nn_2 = cuNearestNeighbors(n_neighbors=1, algorithm="auto").fit(gpu_cloud_2)
    min_distances_1_to_2, _ = nn_2.kneighbors(gpu_cloud_1)
    min_distances_1_to_2 = cp.asnumpy(min_distances_1_to_2).flatten()

    nn_1 = cuNearestNeighbors(n_neighbors=1, algorithm="auto").fit(gpu_cloud_1)
    min_distances_2_to_1, _ = nn_1.kneighbors(gpu_cloud_2)
    min_distances_2_to_1 = cp.asnumpy(min_distances_2_to_1).flatten()

    chamfer_dist = (np.mean(min_distances_1_to_2) + np.mean(min_distances_2_to_1)) / 2

    return chamfer_dist


def _chamfer_kdtree(point_cloud_1, point_cloud_2):
    """Use scipy's cKDTree for efficient Chamfer distance calculation."""
    tree_2 = cKDTree(point_cloud_2)

    min_distances_1_to_2, _ = tree_2.query(point_cloud_1, k=1)

    tree_1 = cKDTree(point_cloud_1)

    min_distances_2_to_1, _ = tree_1.query(point_cloud_2, k=1)

    chamfer_dist = (np.mean(min_distances_1_to_2) + np.mean(min_distances_2_to_1)) / 2

    return chamfer_dist
