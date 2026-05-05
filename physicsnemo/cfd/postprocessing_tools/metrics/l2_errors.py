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

import numpy as np
import pyvista as pv
import torch
from physicsnemo.nn.functional import signed_distance_field

# Match evaluation ``*_l2_numpy`` helpers: relative L2 ||t-p||/||t|| with absolute ||t-p|| when ||t|| ~ 0.
_REL_L2_TRUTH_RTOL = 1e-14


def _relative_l2_normalized(true_field: np.ndarray, pred_field: np.ndarray) -> float:
    """||t - p|| / ||t||; if ||t|| < rtol return ||t - p|| (same rule as ``_l2_pressure_numpy``)."""
    t = np.asarray(true_field, dtype=np.float64).ravel()
    p = np.asarray(pred_field, dtype=np.float64).ravel()
    den = np.linalg.norm(t)
    diff = np.linalg.norm(t - p)
    if den < _REL_L2_TRUTH_RTOL:
        return float(diff)
    return float(diff / den)


def _relative_l2_weighted_sqrt(
    sqrt_weight: np.ndarray, truth: np.ndarray, pred: np.ndarray
) -> float:
    """||w*(t-p)|| / ||w*t|| when ||w*t|| >= rtol else ||w*(t-p)||."""
    w = np.asarray(sqrt_weight, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    wt = (w * t).ravel()
    wdiff = (w * (t - p)).ravel()
    den = np.linalg.norm(wt)
    num = np.linalg.norm(wdiff)
    if den < _REL_L2_TRUTH_RTOL:
        return float(num)
    return float(num / den)


def triangulate_surface_mesh(surface: pv.DataSet) -> pv.PolyData:
    """Triangle-only surface for VTK connectivity backed by indices (SDF / area queries).

    ``PolyData.faces`` layout ``[n, i, j, k, ...]`` is only safe to reshape for uniform
    triangle strips; arbitrary n-gons need triangulation before using :attr:`~pyvista.PolyData.regular_faces`.
    """
    if not isinstance(surface, pv.PolyData):
        surface = surface.extract_surface()
    return surface.triangulate()


def compute_l2_errors(data, true_fields, pred_fields, bounds=None, dtype="point"):
    """Compute L2 error for a given mesh with true and pred fields

    Parameters
    ----------
    data :
        PyVista Dataset
    true_fields :
        List of fields to compute L2 errors for. Should contain the names of true fields.
    pred_fields :
        List of fields to compute L2 errors for. Should contain the names of pred fields.
    bounds :
        Bounds if clipping of the data is required. Bounds must be in following format
        [xmin, xmax, ymin, ymax, zmin, zmax], by default None, which uses entire data
    dtype : str, optional
        Wether to compute drag from cell data or point data, by default "point"

    Returns
    -------
    _type_
        Output dictionary containing L2 errors
    """

    true_fields_list = true_fields
    pred_fields_list = pred_fields

    assert len(true_fields_list) == len(
        pred_fields_list
    ), "True and Pred fields not same"

    # identify vector and scalar quantities
    field_type = {}
    for field in true_fields_list:
        arr = data.get_array(field, preference=dtype)
        if len(arr.shape) == 1:
            field_type[field] = "scalar"
        else:
            field_type[field] = "vector"

    output_dict = {}
    for true, pred in zip(true_fields_list, pred_fields_list):
        true_field = data.get_array(true, preference=dtype)
        pred_field = data.get_array(pred, preference=dtype)
        if bounds is not None:
            points = data.points
            mask = (
                (points[:, 0] >= bounds[0])
                & (points[:, 0] <= bounds[1])
                & (points[:, 1] >= bounds[2])
                & (points[:, 1] <= bounds[3])
                & (points[:, 2] >= bounds[4])
                & (points[:, 2] <= bounds[5])
            )
            true_field = true_field[mask]
            pred_field = pred_field[mask]

        if field_type[true] == "vector":
            # vector quantity
            err_x = _relative_l2_normalized(true_field[:, 0:1], pred_field[:, 0:1])
            err_y = _relative_l2_normalized(true_field[:, 1:2], pred_field[:, 1:2])
            err_z = _relative_l2_normalized(true_field[:, 2:3], pred_field[:, 2:3])

            output_dict[f"{true}_x_l2_error"] = err_x
            output_dict[f"{true}_y_l2_error"] = err_y
            output_dict[f"{true}_z_l2_error"] = err_z
        elif field_type[true] == "scalar":
            # scalar quantity
            err = _relative_l2_normalized(true_field, pred_field)

            output_dict[f"{true}_l2_error"] = err

    return output_dict


def compute_area_weighted_l2_errors(data, true_fields, pred_fields, dtype="point"):
    """Compute L2 error for a given mesh with true and pred fields

    Parameters
    ----------
    data :
        PyVista Dataset
    true_fields :
        List of fields to compute L2 errors for. Should contain the names of true fields.
    pred_fields :
        List of fields to compute L2 errors for. Should contain the names of predicted fields.
    dtype : str, optional
        Wether to compute drag from cell data or point data, by default "point"

    Returns
    -------
    _type_
        Output dictionary containing L2 errors
    """

    if dtype == "cell":
        data = data.compute_cell_sizes()
        areas = data.get_array("Area", preference=dtype)
    elif dtype == "point":
        areas = data.get_array("area", preference=dtype)

    true_fields_list = true_fields
    pred_fields_list = pred_fields

    assert len(true_fields_list) == len(
        pred_fields_list
    ), "True and Pred fields not same"

    # identify vector and scalar quantities
    field_type = {}
    for field in true_fields_list:
        arr = data.get_array(field, preference=dtype)
        if len(arr.shape) == 1:
            field_type[field] = "scalar"
        else:
            field_type[field] = "vector"

    output_dict = {}
    sw_areas = np.sqrt(areas.reshape(-1, 1))

    for true, pred in zip(true_fields_list, pred_fields_list):
        if field_type[true] == "vector":
            ta = data.get_array(true, preference=dtype)
            pa = data.get_array(pred, preference=dtype)
            err_x = _relative_l2_weighted_sqrt(sw_areas, ta[:, 0:1], pa[:, 0:1])
            err_y = _relative_l2_weighted_sqrt(sw_areas, ta[:, 1:2], pa[:, 1:2])
            err_z = _relative_l2_weighted_sqrt(sw_areas, ta[:, 2:3], pa[:, 2:3])

            output_dict[f"{true}_x_area_wt_l2_error"] = err_x
            output_dict[f"{true}_y_area_wt_l2_error"] = err_y
            output_dict[f"{true}_z_area_wt_l2_error"] = err_z
        elif field_type[true] == "scalar":
            # scalar quantity
            ta = data.get_array(true, preference=dtype)
            pa = data.get_array(pred, preference=dtype)
            err = _relative_l2_weighted_sqrt(np.sqrt(areas), ta, pa)

            output_dict[f"{true}_area_wt_l2_error"] = err

    return output_dict


def compute_error_vs_sdf(
    data, true_fields, pred_fields, stl_mesh, bin_edges, bounds=None, dtype="point"
):
    """Compute L2 error vs signed distance field (SDF) for a given mesh with true and pred fields

    This function computes error metrics as a function of distance from a surface defined by an STL mesh.
    The errors are binned based on the signed distance field values and mean errors are computed for each bin.

    Parameters
    ----------
    data :
        PyVista Dataset
    true_fields :
        List of fields to compute L2 errors for. Should contain the names of true fields.
    pred_fields :
        List of fields to compute L2 errors for. Should contain the names of predicted fields.
    stl_mesh :
        PyVista dataset for the STL surface used in SDF evaluation. Triangulated internally
        (``triangulate`` + ``regular_faces``) so n-gons do not misuse ``faces`` layout.
    bin_edges :
        Array defining the bin edges for SDF-based error analysis
    bounds :
        Bounds if clipping of the data is required. Bounds must be in following format
        [xmin, xmax, ymin, ymax, zmin, zmax], by default None, which uses entire data
    dtype : str, optional
        Whether to compute errors from cell data or point data, by default "point"

    Returns
    -------
    dict
        Output dictionary containing L2 error histograms with bin edges and mean errors for each field.
        For vector fields, returns magnitude of error. For scalar fields, returns absolute error.
        Each field entry contains:
        - "bin_edges": List of SDF bin edges
        - "mean_errors": List of mean errors for each SDF bin
    """
    tri = triangulate_surface_mesh(stl_mesh)
    stl_vertices = torch.tensor(np.asarray(tri.points), dtype=torch.float32)
    stl_faces = np.asarray(tri.regular_faces, dtype=np.int64)
    stl_indices = torch.tensor(stl_faces.flatten(), dtype=torch.int32)
    query_points = torch.tensor(np.asarray(data.points), dtype=torch.float32)

    sdf_field, _ = signed_distance_field(
        stl_vertices, stl_indices, query_points, use_sign_winding_number=True
    )
    sdf_field = sdf_field.numpy()

    true_fields_list = true_fields
    pred_fields_list = pred_fields

    assert len(true_fields_list) == len(
        pred_fields_list
    ), "True and Pred fields not same"

    # identify vector and scalar quantities
    field_type = {}
    for field in true_fields_list:
        arr = data.get_array(field, preference=dtype)
        if len(arr.shape) == 1:
            field_type[field] = "scalar"
        else:
            field_type[field] = "vector"

    output_dict = {}
    for true, pred in zip(true_fields_list, pred_fields_list):
        true_field = data.get_array(true, preference=dtype)
        pred_field = data.get_array(pred, preference=dtype)
        if bounds is not None:
            points = data.points
            mask = (
                (points[:, 0] >= bounds[0])
                & (points[:, 0] <= bounds[1])
                & (points[:, 1] >= bounds[2])
                & (points[:, 1] <= bounds[3])
                & (points[:, 2] >= bounds[4])
                & (points[:, 2] <= bounds[5])
            )
            true_field = true_field[mask]
            pred_field = pred_field[mask]
            sdf_sub = sdf_field[mask]
        else:
            sdf_sub = sdf_field

        if field_type[true] == "vector":
            # Compute per-point error magnitude for histogram
            per_point_error = np.linalg.norm(true_field - pred_field, axis=1)

        elif field_type[true] == "scalar":
            # Per-point error for histogram
            per_point_error = np.abs(true_field - pred_field)

        # For each bin, compute the mean error of points in that bin
        num_bins = bin_edges.shape[0] - 1
        bin_indices = np.digitize(sdf_sub, bin_edges) - 1  # -1 to convert to 0-based
        bin_mean_errors = []
        for i in range(num_bins):
            mask = bin_indices == i
            if np.any(mask):
                mean_err = np.mean(per_point_error[mask])
            else:
                mean_err = np.nan  # or 0, or skip, depending on your preference
            bin_mean_errors.append(mean_err)
        # Store in output dict
        output_dict[f"{true}_l2_error_histogram"] = {
            "bin_edges": bin_edges.tolist(),
            "mean_errors": bin_mean_errors,
        }

    return output_dict
