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

"""Build a PyVista comparison mesh with GT and prediction VTK array names for bench metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyvista as pv

from physicsnemo.cfd.evaluation.config import OutputConfig
from physicsnemo.cfd.evaluation.datasets.schema import (
    CanonicalCase,
    FieldDistribution,
    distribution_mean,
)
from physicsnemo.cfd.postprocessing_tools.interpolation.interpolate_mesh_to_pc import (
    interpolate_point_data_to_cell_centers,
)


def _unwrap_prediction_means(predictions: dict[str, Any] | None) -> dict[str, Any]:
    """Return a predictions view with each :class:`FieldDistribution` replaced by its ``mean``.

    Deterministic array values pass through unchanged. Used so dof inference and mesh field
    assignment (which expect plain arrays) work for probabilistic wrappers too.
    """
    if not predictions:
        return {}
    return {k: distribution_mean(v) for k, v in predictions.items()}


def _attach_uq_std_companions(
    mesh: pv.DataSet,
    preference: str,
    predictions: dict[str, Any],
    pairs: tuple[tuple[str, str], ...],
    pred_map: dict[str, str],
    std_map: dict[str, str],
    epi_std_map: dict[str, str],
) -> list[str]:
    """Attach ``*_std`` / ``*_epistemic_std`` arrays for any prediction that is a distribution.

    Names come from the configured std maps, else are auto-derived by suffixing the prediction
    field name with ``"Std"`` / ``"EpistemicStd"``. Missing std channels are skipped silently.
    Returns the list of attached array names so callers can interpolate them to cell dofs
    alongside the mean fields (keeping std on the same dof as its prediction).
    """
    attached: list[str] = []
    for canonical, _ in pairs:
        value = predictions.get(canonical)
        if not isinstance(value, FieldDistribution) or canonical not in pred_map:
            continue
        pred_name = pred_map[canonical]
        if value.std is not None:
            name = std_map.get(canonical) or f"{pred_name}Std"
            _assign_field(mesh, preference, name, value.std)
            attached.append(name)
        if value.epistemic_std is not None:
            name = epi_std_map.get(canonical) or f"{pred_name}EpistemicStd"
            _assign_field(mesh, preference, name, value.epistemic_std)
            attached.append(name)
    return attached


def _infer_surface_preference(
    mesh: pv.PolyData,
    gt: dict[str, Any],
    mesh_type: str,
    predictions: dict[str, Any] | None,
) -> str:
    """
    Choose point vs cell attachment for surface fields.

    Prefer locating arrays in ``ground_truth``, then ``predictions``, and validate against
    ``mesh.n_points`` / ``mesh.n_cells``.
    """

    preds = predictions or {}

    def _collect_ref_from(*sources: dict[str, Any]) -> Any:
        for d in sources:
            for key in ("pressure", "shear_stress"):
                ref = d.get(key)
                if ref is not None:
                    return ref
        return None

    ref = _collect_ref_from(gt, preds)
    if ref is not None:
        a = np.asarray(ref)
        n = int(a.shape[0]) if a.ndim >= 2 else int(a.size)
        if mesh.n_points != mesh.n_cells:
            if n == mesh.n_points:
                return "point"
            if n == mesh.n_cells:
                return "cell"
            raise ValueError(
                f"surface field sample count ({n}) matches neither mesh.n_points ({mesh.n_points}) "
                f"nor mesh.n_cells ({mesh.n_cells}); check GT / prediction dof layout."
            )
        # Ambiguous topology; trust adapter only when dof is explicitly given.
        if mesh_type in ("point", "cell"):
            return mesh_type
        raise ValueError(
            "surface mesh has identical point/cell topology counts while mesh_type is 'unknown'; "
            "cannot infer dof. Set CanonicalCase.mesh_type to 'point' or 'cell', or gt_data_type "
            "so extracted GT includes a dof location."
        )

    # No usable pressure/shear tensors in GT or predictions.
    if mesh_type in ("point", "cell"):
        return mesh_type
    raise ValueError(
        "Cannot infer surface point vs cell dof: mesh_type is 'unknown' while ground_truth has "
        "no usable pressure/shear arrays and predictions lack those tensors. "
        "Provide GT on point or cell dofs, predictions with field arrays, "
        "or adapter mesh_type 'point' / 'cell'."
    )


def _infer_volume_preference(
    mesh: pv.DataSet,
    gt: dict[str, Any],
    predictions: dict[str, Any],
    mesh_type: str,
) -> str:
    """
    Choose point vs cell attachment for volume fields.

    Prefer ``mesh_type`` from the dataset adapter, but validate against array length.
    Uses the first available canonical field among pressure / velocity / turbulent_viscosity,
    preferring ground truth then predictions (GT may be absent in edge cases).
    """
    ref = None
    for key in ("pressure", "velocity", "turbulent_viscosity"):
        if gt:
            ref = gt.get(key)
        if ref is None and predictions:
            ref = predictions.get(key)
        if ref is not None:
            break
    if ref is not None:
        a = np.asarray(ref)
        n = int(a.shape[0]) if a.ndim >= 2 else int(a.size)
        if mesh.n_points != mesh.n_cells:
            if n == mesh.n_points:
                return "point"
            if n == mesh.n_cells:
                return "cell"
            raise ValueError(
                f"volume field sample count ({n}) matches neither mesh.n_points ({mesh.n_points}) "
                f"nor mesh.n_cells ({mesh.n_cells}); check GT / prediction dof layout."
            )
        elif mesh_type in ("point", "cell"):
            return mesh_type
    if mesh_type in ("point", "cell"):
        return mesh_type
    return "cell"


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
            raise ValueError(
                f"Unsupported point array shape for {name!r}: {data.shape}"
            )


def build_comparison_mesh(
    case: CanonicalCase,
    predictions: dict[str, Any],
    output: OutputConfig,
    *,
    mesh_override: pv.DataSet | None = None,
) -> tuple[pv.DataSet, str]:
    """Load case geometry and attach GT / prediction fields for ``physicsnemo.cfd.postprocessing_tools`` metrics.

    Parameters
    ----------
    mesh_override
        If set, use this dataset instead of ``pv.read(case.mesh_path)`` (e.g. tests, in-memory
        pipelines, or :attr:`~physicsnemo.cfd.evaluation.datasets.schema.CanonicalCase.reference_geometry`
        from the adapter). ``case.mesh_path`` is still required on ``CanonicalCase`` for logging
        and fallbacks but is ignored for loading when **override** is provided.

    Returns
    -------
    mesh
        PyVista dataset.
        dtype
        ``\"cell\"`` or ``\"point\"`` for ``compute_l2_errors`` / ``compute_drag_and_lift``.
        For **surface**, inferred from GT and/or prediction array lengths vs ``mesh.n_points`` /
        ``mesh.n_cells`` when unambiguous. If :attr:`~physicsnemo.cfd.evaluation.datasets.schema.CanonicalCase.mesh_type`
        is ``\"unknown\"`` and fields are missing, an error is raised instead of defaulting to cell.
        For **volume**, comparison meshes use **point** dofs (typical unstructured volume layout).
        If ``output.surface_interpolate_point_to_cell_for_metrics`` is true and the mesh used point
        dofs, fields are IDW-interpolated to cell centers and the returned dtype is ``\"cell\"``.
        For **volume**, inferred the same way from GT / predictions vs topology; VTU reference data
        is often cell-centered (``pMean`` on cells).
    """
    if mesh_override is not None:
        # Shallow copy gives independent ``point_data`` / ``cell_data`` containers while sharing
        # the underlying geometry — attaching GT / pred arrays here does not mutate the caller's
        # ``reference_geometry``, and it avoids doubling peak RAM on large volume meshes.
        mesh = mesh_override.copy(deep=False)
    else:
        # Fresh read; no second consumer, no need to copy.
        mesh = pv.read(case.mesh_path)
    gt = case.ground_truth or {}
    # Probabilistic wrappers return FieldDistribution; unwrap to the mean for dof inference and
    # field assignment. Std channels are attached separately below.
    pred_means = _unwrap_prediction_means(predictions)

    if case.inference_domain == "surface":
        gt_map = output.ground_truth_mesh_field_names
        pred_map = output.mesh_field_names
        std_map = output.std_mesh_field_names
        epi_std_map = output.epistemic_std_mesh_field_names
        pairs = (("pressure", "pressure"), ("shear_stress", "shear_stress"))
        if not isinstance(mesh, pv.PolyData):
            mesh = mesh.extract_surface()
        # Do not call point_data_to_cell_data here: it can change n_cells vs the mesh the
        # adapter used when extracting GT, causing "expected N cell values, got ..." mismatches.
        preference = _infer_surface_preference(mesh, gt, case.mesh_type, pred_means)
    elif case.inference_domain == "volume":
        preference = _infer_volume_preference(mesh, gt, pred_means, case.mesh_type)
        gt_map = output.ground_truth_volume_mesh_field_names
        pred_map = output.volume_mesh_field_names
        std_map = output.std_volume_mesh_field_names
        epi_std_map = output.epistemic_std_volume_mesh_field_names
        pairs = (
            ("pressure", "pressure"),
            ("velocity", "velocity"),
            ("turbulent_viscosity", "turbulent_viscosity"),
        )
    else:
        raise ValueError(f"Unknown inference_domain: {case.inference_domain!r}")

    for canonical, _ in pairs:
        if canonical in gt_map and canonical in gt and gt[canonical] is not None:
            _assign_field(mesh, preference, gt_map[canonical], gt[canonical])
        if (
            canonical in pred_map
            and canonical in pred_means
            and pred_means[canonical] is not None
        ):
            _assign_field(mesh, preference, pred_map[canonical], pred_means[canonical])

    std_names = _attach_uq_std_companions(
        mesh, preference, predictions or {}, pairs, pred_map, std_map, epi_std_map
    )

    if (
        case.inference_domain == "surface"
        and output.surface_interpolate_point_to_cell_for_metrics
        and preference == "point"
    ):
        names: list[str] = []
        for canonical, _ in pairs:
            if canonical in gt_map:
                names.append(gt_map[canonical])
            if canonical in pred_map:
                names.append(pred_map[canonical])
        # Interpolate the UQ std companions on the same dof as their mean fields so metrics that
        # read std off cells (e.g. drag_uq) and the saved comparison mesh stay dof-consistent.
        names.extend(std_names)
        names = list(dict.fromkeys(names))
        interpolate_point_data_to_cell_centers(
            mesh,
            names,
            k=output.surface_metrics_idw_k,
            device=output.surface_metrics_idw_device,
        )
        preference = "cell"

    return mesh, preference


def resolve_comparison_mesh_for_metric(
    predictions: dict[str, Any],
    *,
    case: Any,
    comparison_mesh: Any,
    metric_dtype: str | None,
    output: Any,
) -> tuple[Any, str] | tuple[None, None]:
    """Reuse the benchmark engine's comparison mesh or build one from ``case`` + ``output``.

    When the caller already merged mesh + dof label (benchmark path), forwards
    ``(comparison_mesh, metric_dtype)``. Otherwise calls :func:`build_comparison_mesh`, passing
    ``mesh_override=case.reference_geometry`` when the case provides it so adapters may avoid a second
    ``pv.read(case.mesh_path)``.
    """
    if comparison_mesh is not None and metric_dtype is not None:
        return comparison_mesh, metric_dtype
    if case is not None and output is not None:
        ref = getattr(case, "reference_geometry", None)
        return build_comparison_mesh(case, predictions, output, mesh_override=ref)
    return None, None
