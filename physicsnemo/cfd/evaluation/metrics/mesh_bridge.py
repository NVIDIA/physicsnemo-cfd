# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a PyVista comparison mesh with GT and prediction VTK array names for bench metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyvista as pv

from physicsnemo.cfd.evaluation.config import OutputConfig
from physicsnemo.cfd.evaluation.datasets.schema import CanonicalCase


def _assign_field(
    mesh: pv.DataSet,
    preference: str,
    name: str,
    arr: np.ndarray | Any,
) -> None:
    data = np.asarray(arr, dtype=np.float64)
    if preference == "cell":
        n = mesh.n_cells
        if data.ndim == 1:
            if data.size != n:
                raise ValueError(
                    f"Field {name!r}: expected {n} cell values, got shape {data.shape}"
                )
            mesh.cell_data[name] = data
        elif data.ndim == 2 and data.shape[1] == 3:
            if data.shape[0] != n:
                raise ValueError(
                    f"Field {name!r}: expected ({n}, 3) for cells, got {data.shape}"
                )
            mesh.cell_data[name] = data
        else:
            raise ValueError(f"Unsupported cell array shape for {name!r}: {data.shape}")
    else:
        n = mesh.n_points
        if data.ndim == 1:
            if data.size != n:
                raise ValueError(
                    f"Field {name!r}: expected {n} point values, got shape {data.shape}"
                )
            mesh.point_data[name] = data
        elif data.ndim == 2 and data.shape[1] == 3:
            if data.shape[0] != n:
                raise ValueError(
                    f"Field {name!r}: expected ({n}, 3) for points, got {data.shape}"
                )
            mesh.point_data[name] = data
        else:
            raise ValueError(f"Unsupported point array shape for {name!r}: {data.shape}")


def build_comparison_mesh(
    case: CanonicalCase,
    predictions: dict[str, Any],
    output: OutputConfig,
    *,
    mesh_override: pv.DataSet | None = None,
) -> tuple[pv.DataSet, str]:
    """Load case geometry and attach GT / prediction fields for ``physicsnemo.cfd.bench`` metrics.

    Parameters
    ----------
    mesh_override
        If set, use this dataset instead of ``pv.read(case.mesh_path)`` (e.g. tests or in-memory
        pipelines). ``case.mesh_path`` is still accepted on ``CanonicalCase`` but ignored when
        override is provided.

    Returns
    -------
    mesh
        PyVista dataset.
    dtype
        ``\"cell\"`` or ``\"point\"`` for ``compute_l2_errors`` / ``compute_drag_and_lift``.
    """
    if mesh_override is not None:
        mesh = mesh_override.copy(deep=True)
    else:
        mesh = pv.read(case.mesh_path).copy(deep=True)
    gt = case.ground_truth or {}

    if case.inference_domain == "surface":
        mesh = mesh.point_data_to_cell_data(pass_point_data=True)
        preference = "cell"
        gt_map = output.ground_truth_mesh_field_names
        pred_map = output.mesh_field_names
        pairs = (("pressure", "pressure"), ("shear_stress", "shear_stress"))
    elif case.inference_domain == "volume":
        preference = "point"
        gt_map = output.ground_truth_volume_mesh_field_names
        pred_map = output.volume_mesh_field_names
        pairs = (
            ("pressure_volume", "pressure_volume"),
            ("velocity", "velocity"),
            ("turbulent_viscosity", "turbulent_viscosity"),
        )
    else:
        raise ValueError(f"Unknown inference_domain: {case.inference_domain!r}")

    for canonical, _ in pairs:
        if canonical in gt_map and canonical in gt and gt[canonical] is not None:
            _assign_field(mesh, preference, gt_map[canonical], gt[canonical])
        if canonical in pred_map and canonical in predictions and predictions[canonical] is not None:
            _assign_field(mesh, preference, pred_map[canonical], predictions[canonical])

    return mesh, preference
