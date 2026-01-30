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

import pyvista as pv
import numpy as np


def replace_pointset_array_names(
    pointset: pv.PointSet, mapping: dict[str, str]
) -> pv.PointSet:
    """
    Replaces the names of all arrays in a PyVista PointSet.

    Parameters
    ----------
    pointset : pv.PointSet
        The input PointSet.
    mapping : dict of str
        A dictionary mapping old names to new names.

    Returns
    -------
    pv.PointSet
        The PointSet with the array names replaced.
    """
    new_pointset = pv.PointSet(pointset.points)
    for old_name, new_name in mapping.items():
        new_pointset[new_name] = pointset.point_data[old_name]
    return new_pointset


def mask_pointset(pointset: pv.PointSet, mask: np.ndarray) -> pv.PointSet:
    """
    Extracts a subset of points & data from a PyVista PointSet based on a boolean mask.

    Parameters
    ----------
    pointset : pv.PointSet
        The input PointSet.
    mask : np.ndarray
        The boolean mask to apply to the PointSet. Must have the same length as the number
        of points in the PointSet.

    Returns
    -------
    pv.PointSet
        The masked PointSet.

    Raises
    ------
    ValueError
        If the mask length does not match the number of points in the pointset or if the mask
        cannot be cast to a boolean array.
    """
    # Check that the mask is the right length
    if len(mask) != len(pointset.points):
        raise ValueError(
            f"Mask length ({len(mask)}) does not match number of points in pointset ({len(pointset.points)})."
        )

    try:
        mask = mask.astype(bool)
    except ValueError:
        raise ValueError("Mask must be a boolean array, or castable to one.")

    new_pointset = pv.PointSet(pointset.points[mask])
    for name, data in pointset.point_data.items():
        new_pointset.point_data[name] = data[mask]
    return new_pointset


def add_points_to_pointset(
    pointset: pv.PointSet, new_points: np.ndarray, new_data: dict[str, np.ndarray]
) -> pv.PointSet:
    """
    Adds new points and data to a PyVista PointSet.

    Parameters
    ----------
    pointset : pv.PointSet
        The original PointSet.
    new_points : np.ndarray
        The new points to add.
    new_data : dict of str to np.ndarray
        The new data to add. Each key should be a string representing
        the name of the data, and each value should be an array of the same length as new_points.

    Returns
    -------
    pv.PointSet
        The PointSet with the new points and data added.

    Raises
    ------
    ValueError
        If new data does not include all existing data, if the length of new data arrays
        does not match the length of new_points, or if the shape of new data arrays does not
        match the shape of existing data arrays (except for the first dimension).
    """
    # Check that all needed data is provided
    if set(new_data.keys()) != set(pointset.point_data.keys()):
        raise ValueError(
            "New data must include all existing data, and no more.\n"
            f"Existing data: {list(pointset.point_data.keys())}\n"
            f"New data: {list(new_data.keys())}"
        )

    # Check that the new points and new data are consistent
    for name, data in new_data.items():
        if len(data) != len(new_points):
            raise ValueError(
                f"Data array for '{name}' is length {len(data)}, but should be the same length as new_points ({len(new_points)})."
            )

    # Check that the new data is consistent with the existing data (except for length)
    for name, data in new_data.items():
        if name in pointset.point_data:
            expected_shape = pointset.point_data[name].shape[1:]
            if data.shape[1:] != expected_shape:
                raise ValueError(
                    f"Data array for '{name}' has shape {data.shape[1:]}, but should have shape {expected_shape}.\n"
                    "(Note: The first dimension is the number of points.)"
                )

    new_pointset = pv.PointSet(np.concatenate([pointset.points, new_points]))
    for name, data in pointset.point_data.items():
        new_pointset.point_data[name] = np.concatenate([data, new_data[name]])
    return new_pointset
