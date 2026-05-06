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

"""Aggregate volume error visual: resample to structured grid, compute mean/std error, slice and plot."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyvista as pv

from physicsnemo.cfd.postprocessing_tools.interpolation.interpolate_mesh_to_pc import (
    interpolate_mesh_to_pc,
)
from physicsnemo.cfd.postprocessing_tools.visualization.utils import plot_fields
from physicsnemo.cfd.evaluation.config import Config, OutputConfig
from physicsnemo.cfd.evaluation.datasets.progress import log_dataset
from physicsnemo.cfd.evaluation.reports.context_helpers import (
    get_comparison_mesh_for_case,
)
from physicsnemo.cfd.evaluation.reports.registry import register_visual
from physicsnemo.cfd.evaluation.reports.visual_filenames import (
    join_benchmark_visual_segments,
)


def _aggregate_filename_stem(model: str, dataset: str) -> str:
    """``{model}_{dataset}_aggregate`` sanitized for filenames."""
    return join_benchmark_visual_segments(model, dataset, "aggregate")


def _union_axis_aligned_bounds(meshes: Iterable[pv.DataSet]) -> list[float]:
    """Axis-aligned union of ``mesh.bounds`` ``(xmin,xmax,...,zmax)`` (PyVista convention)."""
    u: list[float] | None = None
    for mesh in meshes:
        b = mesh.bounds
        if u is None:
            u = [
                float(b[0]),
                float(b[1]),
                float(b[2]),
                float(b[3]),
                float(b[4]),
                float(b[5]),
            ]
            continue
        u[0] = min(u[0], float(b[0]))
        u[1] = max(u[1], float(b[1]))
        u[2] = min(u[2], float(b[2]))
        u[3] = max(u[3], float(b[3]))
        u[4] = min(u[4], float(b[4]))
        u[5] = max(u[5], float(b[5]))
    if u is None:
        raise ValueError("union bounds requires at least one mesh")
    return u


def _gather_comparison_meshes_for_one_benchmark_run(
    run: dict[str, Any],
    mesh_lookup_run_idx: int,
    case_ids: list[str] | None,
    context: dict[str, Any] | None,
) -> list[pv.DataSet]:
    """Meshes from one benchmark result dict; ``mesh_lookup_run_idx`` indexes ``comparison_meshes_by_run``."""
    meshes: list[pv.DataSet] = []
    if run.get("skipped"):
        return meshes
    for row in run.get("per_case") or []:
        cid = row["case_id"]
        if case_ids is not None and cid not in case_ids:
            continue
        mesh = get_comparison_mesh_for_case(row, cid, mesh_lookup_run_idx, context)
        if mesh is not None:
            meshes.append(mesh)
        else:
            log_dataset(
                "benchmark",
                f"aggregate_volume_errors: no mesh for {cid!r}; skipping case.",
            )
    return meshes


def _axis_linspace(start: float, stop: float, spacing: float) -> np.ndarray:
    """Uniform axis from ``start`` to ``stop`` with deterministic length (avoids ``arange`` FP drift)."""
    if spacing <= 0:
        raise ValueError("voxel_size must be positive")
    lo, hi = (start, stop) if start <= stop else (stop, start)
    extent = hi - lo
    if extent <= 0.0:
        return np.array([lo], dtype=np.float64)
    n = int(round(extent / spacing)) + 1
    n = max(2, n)
    return np.linspace(lo, hi, n, dtype=np.float64)


def _build_structured_grid(bounds: list[float], voxel_size: float) -> pv.StructuredGrid:
    """Create an axis-aligned structured grid from bounds and voxel size."""
    x = _axis_linspace(bounds[0], bounds[1], voxel_size)
    y = _axis_linspace(bounds[2], bounds[3], voxel_size)
    z = _axis_linspace(bounds[4], bounds[5], voxel_size)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return pv.StructuredGrid(xx, yy, zz)


def _resolve_volume_field_pairs(
    output: OutputConfig,
    canonical_keys: list[str] | None,
) -> list[tuple[str, str, str]]:
    """Return (canonical_key, gt_vtk_name, pred_vtk_name) triples for the requested fields."""
    keys = canonical_keys or list(output.ground_truth_volume_mesh_field_names.keys())
    pairs: list[tuple[str, str, str]] = []
    for k in keys:
        gt_name = output.ground_truth_volume_mesh_field_names.get(k)
        pred_name = output.volume_mesh_field_names.get(k)
        if gt_name is None or pred_name is None:
            log_dataset(
                "benchmark",
                f"aggregate_volume_errors: skipping canonical key {k!r} "
                f"(gt={gt_name!r}, pred={pred_name!r})",
            )
            continue
        pairs.append((k, gt_name, pred_name))
    return pairs


def _run_aggregate_pipeline_for_meshes(
    *,
    meshes: list[pv.DataSet],
    vis_dir: Path,
    output_file_stem: str,
    field_pairs: list[tuple[str, str, str]],
    interp_device: str,
    voxel_size: float,
    bounds: list[float] | None,
    y_slice_origin: list[float] | tuple[float, ...],
    z_slice_origin: list[float] | tuple[float, ...],
    save_vtk: bool,
    cmap: str,
    lut: int,
    window_size: list[int] | None,
    plot_vector_components: bool,
    plot_extra_kwargs: dict[str, Any],
) -> None:
    """Interpolate onto grid, accumulate mean/std errors, emit VTK + PNGs for one model×dataset run."""
    if not meshes:
        return

    if bounds is not None:
        effective_bounds = list(bounds)
        if len(effective_bounds) != 6:
            raise ValueError(
                "aggregate_volume_errors bounds must have length 6 "
                "[xmin, xmax, ymin, ymax, zmin, zmax]; "
                f"got length {len(effective_bounds)}"
            )
        log_dataset(
            "benchmark",
            f"aggregate_volume_errors: bounds from config {effective_bounds}",
        )
    else:
        effective_bounds = _union_axis_aligned_bounds(meshes)
        log_dataset(
            "benchmark",
            f"aggregate_volume_errors: bounds mesh union {effective_bounds}",
        )

    all_vtk_names: list[str] = []
    for _, gt_name, pred_name in field_pairs:
        all_vtk_names.extend([gt_name, pred_name])

    error_arrays: dict[str, list[np.ndarray]] = {
        f"{gt_name}_error": [] for _, gt_name, _ in field_pairs
    }

    template_grid: pv.StructuredGrid | None = None
    for mesh in meshes:
        grid = _build_structured_grid(effective_bounds, voxel_size)
        grid = interpolate_mesh_to_pc(
            grid, mesh, all_vtk_names, mesh_dtype="point", device=interp_device
        )

        if template_grid is None:
            template_grid = grid

        for _, gt_name, pred_name in field_pairs:
            error = np.abs(grid.point_data[gt_name] - grid.point_data[pred_name])
            error_arrays[f"{gt_name}_error"].append(error)

    case_count = len(meshes)
    if template_grid is None:
        raise RuntimeError(
            "aggregate_volume_errors: template grid is unset after processing meshes (internal error)."
        )
    log_dataset(
        "benchmark", f"aggregate_volume_errors: aggregating over {case_count} cases."
    )

    fields_to_plot: list[str] = []
    for key, arrays in error_arrays.items():
        stacked = np.stack(arrays, axis=0)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        template_grid.point_data[f"{key}_mean"] = mean
        template_grid.point_data[f"{key}_std"] = std
        fields_to_plot.append(f"{key}_mean")
        fields_to_plot.append(f"{key}_std")

    if save_vtk:
        vtk_path = vis_dir / f"{output_file_stem}_resampled_volume.vtk"
        template_grid.save(str(vtk_path))
        log_dataset("benchmark", f"Wrote {vtk_path}")

    win = window_size or [1280, 3840]

    y_slice = template_grid.slice(normal="y", origin=tuple(y_slice_origin))
    plotter = plot_fields(
        y_slice,
        fields_to_plot,
        plot_vector_components=plot_vector_components,
        view="xz",
        dtype="point",
        cmap=cmap,
        lut=lut,
        window_size=win,
        **plot_extra_kwargs,
    )
    y_png = vis_dir / f"{output_file_stem}_volume_y_slice.png"
    plotter.screenshot(str(y_png))
    plotter.close()
    log_dataset("benchmark", f"Wrote {y_png}")

    z_slice = template_grid.slice(normal="z", origin=tuple(z_slice_origin))
    plotter = plot_fields(
        z_slice,
        fields_to_plot,
        plot_vector_components=plot_vector_components,
        view="xy",
        dtype="point",
        cmap=cmap,
        lut=lut,
        window_size=win,
        **plot_extra_kwargs,
    )
    z_png = vis_dir / f"{output_file_stem}_volume_z_slice.png"
    plotter.screenshot(str(z_png))
    plotter.close()
    log_dataset("benchmark", f"Wrote {z_png}")


def aggregate_volume_errors(
    config: Config,
    results: list[dict[str, Any]],
    output_dir: str,
    *,
    context: dict[str, Any] | None = None,
    case_ids: list[str] | None = None,
    canonical_keys: list[str] | None = None,
    bounds: list[float] | None = None,
    voxel_size: float = 0.03,
    y_slice_origin: list[float] | tuple[float, ...] = (0, 0, 0),
    z_slice_origin: list[float] | tuple[float, ...] = (0, 0, -0.2376),
    save_vtk: bool = True,
    cmap: str = "jet",
    lut: int = 20,
    window_size: list[int] | None = None,
    plot_vector_components: bool = True,
    device: str | None = None,
    **kwargs: Any,
) -> None:
    """Resample volume cases onto a common structured grid, aggregate mean/std errors, and plot slices.

    One VTK + two PNGs are written **per** non-skipped benchmark result (model × dataset).

    When ``config.run.device`` (or explicit ``device``) is CUDA-capable,
    ``interpolate_mesh_to_pc`` performs kNN work on GPU.

    Parameters
    ----------
    canonical_keys : list[str] or None
        Volume field canonical names (e.g. ``["pressure", "velocity"]``).
        Defaults to all keys in ``output.ground_truth_volume_mesh_field_names``.
    bounds : list[float] or None
        ``[xmin, xmax, ymin, ymax, zmin, zmax]`` for the structured grid. When omitted,
        bounds are the axis-aligned **union** of comparison meshes **for each** model×dataset run.
    voxel_size : float
        Grid spacing (metres). Smaller = finer but heavier.
    y_slice_origin, z_slice_origin : tuple
        Origins for the y-normal and z-normal slices.
    save_vtk : bool
        Whether to write resampled VTK for each pair.
    device : str or None
        Passed to ``interpolate_mesh_to_pc``; ``None`` uses ``config.run.device``.
    """
    vis_dir = Path(output_dir) / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)

    interp_device = device if device is not None else config.run.device
    log_dataset(
        "benchmark", f"aggregate_volume_errors: interpolation device={interp_device!r}"
    )

    field_pairs = _resolve_volume_field_pairs(config.output, canonical_keys)
    if not field_pairs:
        log_dataset(
            "benchmark", "aggregate_volume_errors: no valid field pairs; skipping."
        )
        return

    meshes_by_run = (context or {}).get("comparison_meshes_by_run")

    processed = 0
    for run_idx, run in enumerate(results):
        if run.get("skipped"):
            continue

        model = run.get("model") or "model"
        dataset = run.get("dataset") or "dataset"
        file_stem = _aggregate_filename_stem(model, dataset)

        per_run_context = context
        if isinstance(meshes_by_run, list) and run_idx < len(meshes_by_run):
            per_run_context = dict(context or {})
            per_run_context["comparison_meshes_by_run"] = [meshes_by_run[run_idx]]

        meshes = _gather_comparison_meshes_for_one_benchmark_run(
            run, 0, case_ids, per_run_context
        )
        if not meshes:
            log_dataset(
                "benchmark",
                f"aggregate_volume_errors: no meshes for model={model!r} dataset={dataset!r}; skipping.",
            )
            continue

        log_dataset(
            "benchmark",
            f"aggregate_volume_errors: run {processed + 1} file stem {file_stem!r}",
        )
        _run_aggregate_pipeline_for_meshes(
            meshes=meshes,
            vis_dir=vis_dir,
            output_file_stem=file_stem,
            field_pairs=field_pairs,
            interp_device=interp_device,
            voxel_size=voxel_size,
            bounds=bounds,
            y_slice_origin=y_slice_origin,
            z_slice_origin=z_slice_origin,
            save_vtk=save_vtk,
            cmap=cmap,
            lut=lut,
            window_size=window_size,
            plot_vector_components=plot_vector_components,
            plot_extra_kwargs=dict(kwargs),
        )
        processed += 1

    if processed == 0:
        log_dataset(
            "benchmark",
            "aggregate_volume_errors: no benchmark rows produced outputs; skipping.",
        )


def register_aggregate_volume() -> None:
    """Register the ``aggregate_volume_errors`` visual."""
    register_visual("aggregate_volume_errors", aggregate_volume_errors)
