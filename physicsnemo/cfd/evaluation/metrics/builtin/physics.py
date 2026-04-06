# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Continuity and momentum residual metrics (volume) via physicsnemo.cfd.postprocessing_tools.metrics.physics."""

from __future__ import annotations

from typing import Any

import pyvista as pv

from physicsnemo.cfd.postprocessing_tools.metric_registry import register_metric
from physicsnemo.cfd.postprocessing_tools.metrics.l2_errors import compute_l2_errors
from physicsnemo.cfd.postprocessing_tools.metrics.physics import (
    compute_continuity_residuals,
    compute_momentum_residuals,
)
from physicsnemo.cfd.evaluation.metrics.mesh_bridge import build_comparison_mesh


def _resolve_mesh(
    predictions: dict[str, Any],
    *,
    case: Any,
    comparison_mesh: Any,
    metric_dtype: str | None,
    output: Any,
) -> tuple[pv.DataSet, str] | tuple[None, None]:
    if comparison_mesh is not None and metric_dtype is not None:
        return comparison_mesh, metric_dtype
    if case is not None and output is not None:
        return build_comparison_mesh(case, predictions, output)
    return None, None


def continuity_residual_l2(
    ground_truth: dict,
    predictions: dict,
    *,
    case: Any = None,
    comparison_mesh: Any = None,
    metric_dtype: str | None = None,
    output: Any = None,
    true_velocity_field: str | None = None,
    predicted_velocity_field: str | None = None,
    device: str = "cpu",
    **_: object,
) -> dict[str, float]:
    mesh, _ = _resolve_mesh(
        predictions, case=case, comparison_mesh=comparison_mesh, metric_dtype=metric_dtype, output=output
    )
    if mesh is None or output is None:
        return {"Continuity_l2_error": float("nan")}
    gtn = true_velocity_field or output.ground_truth_volume_mesh_field_names["velocity"]
    prn = predicted_velocity_field or output.volume_mesh_field_names["velocity"]
    try:
        m = mesh.copy(deep=True)
        m = compute_continuity_residuals(m, gtn, prn, device=device)
        return compute_l2_errors(m, ["Continuity"], ["ContinuityPred"], dtype="point")
    except Exception:
        return {"Continuity_l2_error": float("nan")}


def momentum_residual_l2(
    ground_truth: dict,
    predictions: dict,
    *,
    case: Any = None,
    comparison_mesh: Any = None,
    metric_dtype: str | None = None,
    output: Any = None,
    true_velocity_field: str | None = None,
    predicted_velocity_field: str | None = None,
    true_pressure_field: str | None = None,
    predicted_pressure_field: str | None = None,
    true_nu_field: str | None = None,
    predicted_nu_field: str | None = None,
    nu: float = 1.5e-5,
    rho: float = 1.0,
    device: str = "cpu",
    **_: object,
) -> dict[str, float]:
    mesh, _ = _resolve_mesh(
        predictions, case=case, comparison_mesh=comparison_mesh, metric_dtype=metric_dtype, output=output
    )
    if mesh is None or output is None:
        return {"Momentum_x_l2_error": float("nan")}
    uv = true_velocity_field or output.ground_truth_volume_mesh_field_names["velocity"]
    pv = predicted_velocity_field or output.volume_mesh_field_names["velocity"]
    tp = true_pressure_field or output.ground_truth_volume_mesh_field_names["pressure_volume"]
    pp = predicted_pressure_field or output.volume_mesh_field_names["pressure_volume"]
    tn = true_nu_field or output.ground_truth_volume_mesh_field_names["turbulent_viscosity"]
    pn = predicted_nu_field or output.volume_mesh_field_names["turbulent_viscosity"]
    try:
        m = mesh.copy(deep=True)
        m = compute_momentum_residuals(
            m,
            true_velocity_field=uv,
            predicted_velocity_field=pv,
            true_pressure_field=tp,
            predicted_pressure_field=pp,
            true_nu_field=tn,
            predicted_nu_field=pn,
            nu=nu,
            rho=rho,
            device=device,
        )
        return compute_l2_errors(m, ["Momentum"], ["MomentumPred"], dtype="point")
    except Exception:
        return {"momentum_residual_failed": float("nan")}


def register_physics_metrics() -> None:
    register_metric("continuity_residual_l2", continuity_residual_l2)
    register_metric("momentum_residual_l2", momentum_residual_l2)
