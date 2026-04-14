# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate volume error visual: resample to structured grid, compute mean/std error, slice and plot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from physicsnemo.cfd.postprocessing_tools.interpolation.interpolate_mesh_to_pc import (
    interpolate_mesh_to_pc,
)
from physicsnemo.cfd.postprocessing_tools.visualization.utils import plot_fields
from physicsnemo.cfd.evaluation.config import Config, OutputConfig
from physicsnemo.cfd.evaluation.datasets.progress import log_dataset
from physicsnemo.cfd.evaluation.reports.context_helpers import get_comparison_mesh_for_case
from physicsnemo.cfd.evaluation.reports.registry import register_visual

DEFAULT_BOUNDS = [-3.5, 8.5, -2.25, 2.25, -0.32, 3.00]


def _build_structured_grid(
    bounds: list[float], voxel_size: float
) -> pv.StructuredGrid:
    """Create an axis-aligned structured grid from bounds and voxel size."""
    x = np.arange(bounds[0], bounds[1] + voxel_size, voxel_size)
    y = np.arange(bounds[2], bounds[3] + voxel_size, voxel_size)
    z = np.arange(bounds[4], bounds[5] + voxel_size, voxel_size)
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


def _interpolation_device(run_device: str) -> str:
    """Map ``config.run.device`` to a device string for ``interpolate_mesh_to_pc``."""
    return run_device


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

    For each case's comparison mesh, fields are interpolated onto a regular voxel
    grid.  Point-wise ``|GT - pred|`` is computed per field, then mean and std are
    taken across cases.  The aggregate grid is optionally saved as VTK and two
    slices (y-normal and z-normal) are rendered via ``plot_fields``.

    When a CUDA device is available (via ``config.run.device`` or the explicit
    ``device`` kwarg), kNN interpolation uses cuML/CuPy for GPU acceleration.

    Parameters
    ----------
    canonical_keys : list[str] or None
        Volume field canonical names (e.g. ``["pressure", "velocity"]``).
        Defaults to all keys in ``output.ground_truth_volume_mesh_field_names``.
    bounds : list[float] or None
        ``[xmin, xmax, ymin, ymax, zmin, zmax]`` for the structured grid.
    voxel_size : float
        Grid spacing (metres). Smaller = finer but heavier.
    y_slice_origin, z_slice_origin : tuple
        Origins for the y-normal and z-normal slices.
    save_vtk : bool
        Whether to write ``aggregate_resampled_volume.vtk``.
    device : str or None
        ``"cpu"`` or ``"gpu"``.  When ``None`` (default), inferred from
        ``config.run.device`` — CUDA devices map to ``"gpu"``.
    """
    vis_dir = Path(output_dir) / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)

    interp_device = device if device is not None else _interpolation_device(config.run.device)
    log_dataset("benchmark", f"aggregate_volume_errors: interpolation device={interp_device!r}")
    effective_bounds = bounds if bounds is not None else list(DEFAULT_BOUNDS)
    field_pairs = _resolve_volume_field_pairs(config.output, canonical_keys)
    if not field_pairs:
        log_dataset("benchmark", "aggregate_volume_errors: no valid field pairs; skipping.")
        return

    all_vtk_names = []
    for _, gt_name, pred_name in field_pairs:
        all_vtk_names.extend([gt_name, pred_name])

    error_arrays: dict[str, list[np.ndarray]] = {
        f"{gt_name}_error": [] for _, gt_name, _ in field_pairs
    }

    template_grid: pv.StructuredGrid | None = None
    case_count = 0

    for run_idx, run in enumerate(results):
        if run.get("skipped"):
            continue
        for row in run.get("per_case") or []:
            cid = row["case_id"]
            if case_ids is not None and cid not in case_ids:
                continue

            mesh = get_comparison_mesh_for_case(row, cid, run_idx, context)
            if mesh is None:
                log_dataset(
                    "benchmark",
                    f"aggregate_volume_errors: no mesh for {cid!r}; skipping case.",
                )
                continue

            grid = _build_structured_grid(effective_bounds, voxel_size)
            grid = interpolate_mesh_to_pc(grid, mesh, all_vtk_names, mesh_dtype="point", device=interp_device)

            if template_grid is None:
                template_grid = grid

            for _, gt_name, pred_name in field_pairs:
                error = np.abs(
                    grid.point_data[gt_name] - grid.point_data[pred_name]
                )
                error_arrays[f"{gt_name}_error"].append(error)

            case_count += 1

    if template_grid is None or case_count == 0:
        log_dataset("benchmark", "aggregate_volume_errors: no cases processed; skipping.")
        return

    log_dataset("benchmark", f"aggregate_volume_errors: aggregating over {case_count} cases.")

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
        vtk_path = vis_dir / "aggregate_resampled_volume.vtk"
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
        **kwargs,
    )
    y_png = vis_dir / "aggregate_volume_y_slice.png"
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
        **kwargs,
    )
    z_png = vis_dir / "aggregate_volume_z_slice.png"
    plotter.screenshot(str(z_png))
    plotter.close()
    log_dataset("benchmark", f"Wrote {z_png}")


def register_aggregate_volume() -> None:
    register_visual("aggregate_volume_errors", aggregate_volume_errors)
