# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Continuity and momentum residual metrics (volume) via physicsnemo.cfd.postprocessing_tools.metrics.physics."""

from __future__ import annotations

import logging
import traceback
from typing import Any, Callable, TypeVar

from physicsnemo.cfd.postprocessing_tools.metric_registry import register_metric
from physicsnemo.cfd.postprocessing_tools.metrics.l2_errors import compute_l2_errors
from physicsnemo.cfd.postprocessing_tools.metrics.physics import (
    compute_continuity_residuals,
    compute_momentum_residuals,
)
from physicsnemo.cfd.evaluation.metrics.mesh_bridge import resolve_comparison_mesh_for_metric
from physicsnemo.cfd.evaluation.metrics.metric_exceptions import RECOVERABLE_MESH_METRIC_ERRORS

_LOG = logging.getLogger(__name__)
TPhys = TypeVar("TPhys", bound=dict[str, float])


def _nan_momentum_component_l2_error() -> dict[str, float]:
    """Aligned with compute_l2_errors vector keys for field name ``Momentum``."""
    nan = float("nan")
    return {
        "Momentum_x_l2_error": nan,
        "Momentum_y_l2_error": nan,
        "Momentum_z_l2_error": nan,
    }


def _safe_physics_residual(
    metric_name: str,
    compute: Callable[[], TPhys],
    fallback: TPhys,
) -> TPhys:
    """Mesh + residual + L2 path; failures in :data:`RECOVERABLE_MESH_METRIC_ERRORS` yield NaNs with traceback log."""
    try:
        return compute()
    except RECOVERABLE_MESH_METRIC_ERRORS:
        _LOG.warning(
            "%s failed (recoverable; returning fallback result):\n%s",
            metric_name,
            traceback.format_exc(),
        )
        return fallback


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
    mesh, _ = resolve_comparison_mesh_for_metric(
        predictions, case=case, comparison_mesh=comparison_mesh, metric_dtype=metric_dtype, output=output
    )
    if mesh is None or output is None:
        return {"Continuity_l2_error": float("nan")}

    def _run() -> dict[str, float]:
        gt_velocity_name = true_velocity_field or output.ground_truth_volume_mesh_field_names["velocity"]
        pred_velocity_name = predicted_velocity_field or output.volume_mesh_field_names["velocity"]
        m = mesh.copy(deep=True)
        m = compute_continuity_residuals(m, gt_velocity_name, pred_velocity_name, device=device)
        return compute_l2_errors(m, ["Continuity"], ["ContinuityPred"], dtype="point")

    return _safe_physics_residual(
        "continuity_residual_l2",
        _run,
        {"Continuity_l2_error": float("nan")},
    )


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
    mesh, _ = resolve_comparison_mesh_for_metric(
        predictions, case=case, comparison_mesh=comparison_mesh, metric_dtype=metric_dtype, output=output
    )
    if mesh is None or output is None:
        return _nan_momentum_component_l2_error()

    def _run() -> dict[str, float]:
        gt_velocity_name = true_velocity_field or output.ground_truth_volume_mesh_field_names["velocity"]
        pred_velocity_name = predicted_velocity_field or output.volume_mesh_field_names["velocity"]
        gt_pressure_name = true_pressure_field or output.ground_truth_volume_mesh_field_names["pressure"]
        pred_pressure_name = predicted_pressure_field or output.volume_mesh_field_names["pressure"]
        gt_nu_name = true_nu_field or output.ground_truth_volume_mesh_field_names["turbulent_viscosity"]
        pred_nu_name = predicted_nu_field or output.volume_mesh_field_names["turbulent_viscosity"]
        m = mesh.copy(deep=True)
        m = compute_momentum_residuals(
            m,
            true_velocity_field=gt_velocity_name,
            predicted_velocity_field=pred_velocity_name,
            true_pressure_field=gt_pressure_name,
            predicted_pressure_field=pred_pressure_name,
            true_nu_field=gt_nu_name,
            predicted_nu_field=pred_nu_name,
            nu=nu,
            rho=rho,
            device=device,
        )
        return compute_l2_errors(m, ["Momentum"], ["MomentumPred"], dtype="point")

    return _safe_physics_residual(
        "momentum_residual_l2",
        _run,
        _nan_momentum_component_l2_error(),
    )


def register_physics_metrics() -> None:
    register_metric("continuity_residual_l2", continuity_residual_l2, domain="volume")
    register_metric("momentum_residual_l2", momentum_residual_l2, domain="volume")
