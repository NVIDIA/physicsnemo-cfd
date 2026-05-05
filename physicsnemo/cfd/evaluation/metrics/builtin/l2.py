# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L2-style metrics delegating to physicsnemo.cfd.postprocessing_tools.metrics.l2_errors."""

from __future__ import annotations

import logging
import traceback
from typing import Any, Callable, TypeVar

import numpy as np

from physicsnemo.cfd.postprocessing_tools.metric_registry import register_metric
from physicsnemo.cfd.postprocessing_tools.metrics.l2_errors import (
    _relative_l2_normalized,
    compute_area_weighted_l2_errors,
    compute_l2_errors,
)
from physicsnemo.cfd.evaluation.metrics.metric_exceptions import (
    RECOVERABLE_MESH_METRIC_ERRORS,
)
from physicsnemo.cfd.evaluation.metrics.mesh_bridge import (
    resolve_comparison_mesh_for_metric,
)

_LOG = logging.getLogger(__name__)
T = TypeVar("T")


def _safe_l2(metric_name: str, compute: Callable[[], T], fallback: T) -> T:
    """Run mesh-based ``compute_l2_errors`` path; recoverable failures yield NaNs with traceback log."""
    try:
        return compute()
    except RECOVERABLE_MESH_METRIC_ERRORS:
        _LOG.warning(
            "%s failed (recoverable; returning fallback result):\n%s",
            metric_name,
            traceback.format_exc(),
        )
        return fallback


def _l2_pressure_numpy(
    ground_truth: dict,
    predictions: dict,
    mask: np.ndarray | None = None,
    **_: object,
) -> float:
    gt = np.asarray(ground_truth.get("pressure", []), dtype=np.float64).flatten()
    pred = np.asarray(predictions.get("pressure", []), dtype=np.float64).flatten()
    if gt.size == 0 or pred.size == 0 or gt.shape != pred.shape:
        return float("nan")
    if mask is not None:
        m = np.asarray(mask).flatten()
        gt = gt[m]
        pred = pred[m]
    return _relative_l2_normalized(gt, pred)


def _l2_shear_numpy(
    ground_truth: dict,
    predictions: dict,
    mask: np.ndarray | None = None,
    **_: object,
) -> float:
    gt = np.asarray(ground_truth.get("shear_stress", []), dtype=np.float64)
    pred = np.asarray(predictions.get("shear_stress", []), dtype=np.float64)
    if gt.size == 0 or pred.size == 0 or gt.shape != pred.shape:
        return float("nan")
    gt = gt.flatten()
    pred = pred.flatten()
    if mask is not None:
        m = np.asarray(mask).flatten()
        if m.dtype == bool and m.size * 3 == gt.size:
            gt = gt.reshape(-1, 3)[m].flatten()
            pred = pred.reshape(-1, 3)[m].flatten()
        else:
            gt = gt[m]
            pred = pred[m]
    return _relative_l2_normalized(gt, pred)


# ---------------------------------------------------------------------------
# Surface pressure L2
# ---------------------------------------------------------------------------


def l2_pressure_surface(
    ground_truth: dict,
    predictions: dict,
    *,
    case: Any = None,
    comparison_mesh: Any = None,
    metric_dtype: str | None = None,
    output: Any = None,
    mask: np.ndarray | None = None,
    **_: object,
) -> float:
    """Relative L2 error of surface pressure (mesh-aware when ``comparison_mesh`` is available)."""
    mesh, dtype = resolve_comparison_mesh_for_metric(
        predictions,
        case=case,
        comparison_mesh=comparison_mesh,
        metric_dtype=metric_dtype,
        output=output,
    )
    if mesh is None or output is None:
        return _l2_pressure_numpy(ground_truth, predictions, mask=mask)
    gtn = output.ground_truth_mesh_field_names["pressure"]
    prn = output.mesh_field_names["pressure"]

    def _compute() -> float:
        d = compute_l2_errors(mesh, [gtn], [prn], dtype=dtype)
        key = f"{gtn}_l2_error"
        return float(d[key]) if key in d else float("nan")

    return _safe_l2("l2_pressure_surface", _compute, float("nan"))


# Same object as l2_pressure_surface; allows ``from ... import l2_pressure``.
# The registry string "l2_pressure" also binds l2_pressure_volume for volume runs
# (resolved by inference domain). Import l2_pressure_volume for explicit volume L2 calls.
l2_pressure = l2_pressure_surface


# ---------------------------------------------------------------------------
# Volume pressure L2
# ---------------------------------------------------------------------------


def l2_pressure_volume(
    ground_truth: dict,
    predictions: dict,
    *,
    case: Any = None,
    comparison_mesh: Any = None,
    metric_dtype: str | None = None,
    output: Any = None,
    gt_key: str = "pressure",
    pred_key: str | None = None,
    mask: np.ndarray | None = None,
    **_: object,
) -> float:
    """Relative L2 error of volume pressure (mesh-aware when ``comparison_mesh`` is available)."""
    mesh, dtype = resolve_comparison_mesh_for_metric(
        predictions,
        case=case,
        comparison_mesh=comparison_mesh,
        metric_dtype=metric_dtype,
        output=output,
    )
    pk = pred_key if pred_key is not None else gt_key
    if mesh is None or output is None:
        gt = np.asarray(ground_truth.get(gt_key, []), dtype=np.float64).flatten()
        pred = np.asarray(predictions.get(pk, []), dtype=np.float64).flatten()
        if gt.size == 0 or pred.size == 0 or gt.shape != pred.shape:
            return float("nan")
        if mask is not None:
            m = np.asarray(mask).flatten()
            gt = gt[m]
            pred = pred[m]
        return _relative_l2_normalized(gt, pred)
    gtn = output.ground_truth_volume_mesh_field_names[gt_key]
    prn = output.volume_mesh_field_names[pk]

    def _compute() -> float:
        d = compute_l2_errors(mesh, [gtn], [prn], dtype=dtype)
        key = f"{gtn}_l2_error"
        return float(d[key]) if key in d else float("nan")

    return _safe_l2("l2_pressure_volume", _compute, float("nan"))


# ---------------------------------------------------------------------------


def l2_shear_stress(
    ground_truth: dict,
    predictions: dict,
    *,
    case: Any = None,
    comparison_mesh: Any = None,
    metric_dtype: str | None = None,
    output: Any = None,
    mask: np.ndarray | None = None,
    **_: object,
) -> dict[str, float]:
    """Relative L2 error of surface shear stress (vector); returns ``{component: value}`` dict."""
    mesh, dtype = resolve_comparison_mesh_for_metric(
        predictions,
        case=case,
        comparison_mesh=comparison_mesh,
        metric_dtype=metric_dtype,
        output=output,
    )
    if mesh is None or output is None:
        v = _l2_shear_numpy(ground_truth, predictions, mask=mask)
        return {"magnitude": v}
    gtn = output.ground_truth_mesh_field_names["shear_stress"]
    prn = output.mesh_field_names["shear_stress"]

    def _compute() -> dict[str, float]:
        return compute_l2_errors(mesh, [gtn], [prn], dtype=dtype)

    return _safe_l2("l2_shear_stress", _compute, {"magnitude": float("nan")})


# ---------------------------------------------------------------------------
# Area-weighted L2 pressure (surface)
# ---------------------------------------------------------------------------


def l2_pressure_area_weighted(
    ground_truth: dict,
    predictions: dict,
    *,
    case: Any = None,
    comparison_mesh: Any = None,
    metric_dtype: str | None = None,
    output: Any = None,
    **_: object,
) -> float:
    """Area-weighted relative L2 error of surface pressure on the comparison mesh."""
    mesh, dtype = resolve_comparison_mesh_for_metric(
        predictions,
        case=case,
        comparison_mesh=comparison_mesh,
        metric_dtype=metric_dtype,
        output=output,
    )
    if mesh is None or output is None:
        return float("nan")
    gtn = output.ground_truth_mesh_field_names["pressure"]
    prn = output.mesh_field_names["pressure"]

    def _compute() -> float:
        d = compute_area_weighted_l2_errors(mesh, [gtn], [prn], dtype=dtype)
        key = f"{gtn}_area_wt_l2_error"
        return float(d[key]) if key in d else float("nan")

    return _safe_l2("l2_pressure_area_weighted", _compute, float("nan"))


# ---------------------------------------------------------------------------
# Velocity L2 (volume)
# ---------------------------------------------------------------------------


def l2_velocity(
    ground_truth: dict,
    predictions: dict,
    *,
    case: Any = None,
    comparison_mesh: Any = None,
    metric_dtype: str | None = None,
    output: Any = None,
    **_: object,
) -> dict[str, float]:
    """Relative L2 error of volume velocity (vector); returns ``{component: value}`` dict."""
    mesh, dtype = resolve_comparison_mesh_for_metric(
        predictions,
        case=case,
        comparison_mesh=comparison_mesh,
        metric_dtype=metric_dtype,
        output=output,
    )
    if mesh is None or output is None:
        gt = np.asarray(ground_truth.get("velocity", []), dtype=np.float64)
        pred = np.asarray(predictions.get("velocity", []), dtype=np.float64)
        if gt.size == 0 or pred.size == 0 or gt.shape != pred.shape:
            return {"magnitude": float("nan")}
        gt = gt.flatten()
        pred = pred.flatten()
        return {"magnitude": _relative_l2_normalized(gt, pred)}
    gtn = output.ground_truth_volume_mesh_field_names["velocity"]
    prn = output.volume_mesh_field_names["velocity"]

    def _compute() -> dict[str, float]:
        return compute_l2_errors(mesh, [gtn], [prn], dtype=dtype)

    return _safe_l2("l2_velocity", _compute, {"magnitude": float("nan")})


# ---------------------------------------------------------------------------
# Turbulent viscosity L2 (volume)
# ---------------------------------------------------------------------------


def l2_turbulent_viscosity(
    ground_truth: dict,
    predictions: dict,
    *,
    case: Any = None,
    comparison_mesh: Any = None,
    metric_dtype: str | None = None,
    output: Any = None,
    gt_key: str = "turbulent_viscosity",
    pred_key: str | None = None,
    mask: np.ndarray | None = None,
    **_: object,
) -> float:
    """Relative L2 error of volume turbulent viscosity (mesh-aware when available)."""
    mesh, dtype = resolve_comparison_mesh_for_metric(
        predictions,
        case=case,
        comparison_mesh=comparison_mesh,
        metric_dtype=metric_dtype,
        output=output,
    )
    pk = pred_key if pred_key is not None else gt_key
    if mesh is None or output is None:
        gt = np.asarray(ground_truth.get(gt_key, []), dtype=np.float64).flatten()
        pred = np.asarray(predictions.get(pk, []), dtype=np.float64).flatten()
        if gt.size == 0 or pred.size == 0 or gt.shape != pred.shape:
            return float("nan")
        if mask is not None:
            m = np.asarray(mask).flatten()
            gt = gt[m]
            pred = pred[m]
        return _relative_l2_normalized(gt, pred)
    gtn = output.ground_truth_volume_mesh_field_names[gt_key]
    prn = output.volume_mesh_field_names[pk]

    def _compute() -> float:
        d = compute_l2_errors(mesh, [gtn], [prn], dtype=dtype)
        key = f"{gtn}_l2_error"
        return float(d[key]) if key in d else float("nan")

    return _safe_l2("l2_turbulent_viscosity", _compute, float("nan"))


# ---------------------------------------------------------------------------


def register_l2_metrics() -> None:
    """Register the built-in surface and volume L2 metrics with the metric registry."""
    register_metric("l2_pressure", l2_pressure_surface, domain="surface")
    register_metric("l2_shear_stress", l2_shear_stress, domain="surface")
    register_metric(
        "l2_pressure_area_weighted", l2_pressure_area_weighted, domain="surface"
    )
    register_metric("area_wt_l2_pressure", l2_pressure_area_weighted, domain="surface")

    # Volume metrics
    register_metric("l2_pressure", l2_pressure_volume, domain="volume")
    register_metric("l2_velocity", l2_velocity, domain="volume")
    register_metric("l2_turbulent_viscosity", l2_turbulent_viscosity, domain="volume")
