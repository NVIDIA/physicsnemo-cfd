# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Line plot visual wrapping ``physicsnemo.cfd.postprocessing_tools.visualization.utils.plot_line``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pyvista as pv

from physicsnemo.cfd.postprocessing_tools.visualization.utils import plot_line
from physicsnemo.cfd.evaluation.config import Config, OutputConfig
from physicsnemo.cfd.evaluation.datasets.progress import log_dataset
from physicsnemo.cfd.evaluation.reports.context_helpers import get_comparison_mesh_for_case
from physicsnemo.cfd.evaluation.reports.registry import register_visual


def _resolve_gt_pred_fields(output: OutputConfig, canonical_key: str) -> tuple[str, str]:
    """Surface or volume VTK names for a canonical key."""
    if canonical_key in output.ground_truth_mesh_field_names and canonical_key in output.mesh_field_names:
        return (
            output.ground_truth_mesh_field_names[canonical_key],
            output.mesh_field_names[canonical_key],
        )
    if canonical_key in output.ground_truth_volume_mesh_field_names and canonical_key in output.volume_mesh_field_names:
        return (
            output.ground_truth_volume_mesh_field_names[canonical_key],
            output.volume_mesh_field_names[canonical_key],
        )
    raise KeyError(
        f"Canonical key {canonical_key!r} not found in surface or volume output field maps"
    )


def _comparison_mesh_to_line_polydata(mesh: pv.DataSet) -> pv.PolyData:
    """One point per cell with cell fields on points (for ``plot_line`` single-line branch)."""
    if mesh.n_cells > 0:
        return mesh.cell_centers(vertex=False)
    return mesh


def _combine_if_multiblock(ds: pv.DataSet) -> pv.DataSet:
    if isinstance(ds, pv.MultiBlock):
        if ds.n_blocks == 0:
            return pv.PolyData()
        try:
            return ds.combine(merge_points=False)
        except Exception:
            return ds.get_block(0)
    return ds


def bench_style_surface_centerlines(
    mesh: pv.DataSet,
    *,
    y_slice_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    z_clip: float = 0.4,
) -> tuple[pv.PolyData, pv.PolyData]:
    """DrivAer-style top/bottom strips: ``y`` slice at ``y_slice_origin``, split at ``z = z_clip``.

    Matches ``workflows/deprecated/bench_example/utils.py`` (surface): top = ``z`` above the clip plane,
    bottom = below.
    """
    slice_y = mesh.slice(normal="y", origin=y_slice_origin)
    slice_y = _combine_if_multiblock(slice_y)
    if slice_y.n_cells == 0 and slice_y.n_points == 0:
        return pv.PolyData(), pv.PolyData()
    top = slice_y.clip(normal="z", origin=(0.0, 0.0, z_clip), invert=False)
    bottom = slice_y.clip(normal="z", origin=(0.0, 0.0, z_clip), invert=True)
    top = _combine_if_multiblock(top)
    bottom = _combine_if_multiblock(bottom)
    return top, bottom


def _mesh_for_plot_line_centerline(poly: pv.DataSet) -> pv.PolyData | None:
    """``plot_line`` single branch expects point_data for scalar/vector fields."""
    if poly.n_cells == 0 and poly.n_points == 0:
        return None
    if poly.n_cells > 0:
        out = poly.cell_data_to_point_data(pass_cell_data=True)
    else:
        out = poly
    return out if out.n_points > 0 else None


def line_plot_centerlines(
    config: Config,
    results: list[dict[str, Any]],
    output_dir: str,
    *,
    context: dict[str, Any] | None = None,
    case_ids: list[str] | None = None,
    canonical_key: str = "pressure",
    plot_coord: str = "x",
    normalize_factor: float = 1.0,
    coord_trim: tuple[float | None, float | None] | None = None,
    field_trim: tuple[float | None, float | None] | None = None,
    flip: bool = False,
    y_slice_origin: list[float] | None = None,
    z_clip: float = 0.4,
    **kwargs: Any,
) -> None:
    """GT vs pred line plots on **top** and **bottom** surface centerline strips (legacy bench_example style).

    Geometry: slice with normal ``y`` through ``y_slice_origin`` (default origin), then clip at
    ``z = z_clip`` to separate upper (``top``) and lower (``bottom``) surface lines along the vehicle.

    Surface (``metric_dtype`` cell) comparison meshes only. Writes two PNGs per case:
    ``*_line_centerline_top_*`` and ``*_line_centerline_bottom_*``.
    """
    out = Path(output_dir) / "visuals"
    out.mkdir(parents=True, exist_ok=True)
    output: OutputConfig = config.output
    field_true, field_pred = _resolve_gt_pred_fields(output, canonical_key)

    yo = tuple(y_slice_origin) if y_slice_origin is not None and len(y_slice_origin) == 3 else (0.0, 0.0, 0.0)

    ct_raw = coord_trim if coord_trim is not None else (None, None)
    ft_raw = field_trim if field_trim is not None else (None, None)
    ct = (ct_raw[0], ct_raw[1]) if isinstance(ct_raw, (list, tuple)) and len(ct_raw) == 2 else (None, None)
    ft = (ft_raw[0], ft_raw[1]) if isinstance(ft_raw, (list, tuple)) and len(ft_raw) == 2 else (None, None)

    for run_idx, run in enumerate(results):
        if run.get("skipped"):
            continue
        model = run["model"]
        dataset = run["dataset"]
        for row in run.get("per_case") or []:
            cid = row["case_id"]
            if case_ids is not None and cid not in case_ids:
                continue
            if row.get("metric_dtype") != "cell":
                log_dataset(
                    "benchmark",
                    f"Skip line_plot_centerlines for {cid!r}: surface (cell) comparison mesh only",
                )
                continue
            mesh = get_comparison_mesh_for_case(row, cid, run_idx, context)
            if mesh is None:
                log_dataset(
                    "benchmark",
                    f"Skip line_plot_centerlines for {cid!r}: no comparison mesh",
                )
                continue
            try:
                top_raw, bottom_raw = bench_style_surface_centerlines(
                    mesh, y_slice_origin=yo, z_clip=z_clip
                )
            except Exception as ex:
                log_dataset("benchmark", f"line_plot_centerlines slice failed for {cid!r}: {ex}")
                continue

            for label, strip in (("top", top_raw), ("bottom", bottom_raw)):
                line_pd = _mesh_for_plot_line_centerline(strip)
                if line_pd is None:
                    log_dataset(
                        "benchmark",
                        f"Skip line_plot_centerlines {label} for {cid!r}: empty slice",
                    )
                    continue
                if field_true not in line_pd.point_data or field_pred not in line_pd.point_data:
                    log_dataset(
                        "benchmark",
                        f"Skip line_plot_centerlines {label} for {cid!r}: missing {field_true!r} or {field_pred!r} on slice",
                    )
                    continue
                fig = plot_line(
                    line_pd,
                    plot_coord=plot_coord,
                    field_true=field_true,
                    field_pred=field_pred,
                    normalize_factor=normalize_factor,
                    coord_trim=ct,
                    field_trim=ft,
                    flip=flip,
                    **kwargs,
                )
                safe = (
                    f"{model}_{dataset}_{cid}_line_centerline_{label}_{canonical_key}_{plot_coord}.png".replace(
                        "/", "_"
                    )
                )
                out_png = out / safe
                fig.savefig(str(out_png), bbox_inches="tight", dpi=150)
                plt.close(fig)
                log_dataset("benchmark", f"Wrote centerline line plot {out_png}")


def line_plot(
    config: Config,
    results: list[dict[str, Any]],
    output_dir: str,
    *,
    context: dict[str, Any] | None = None,
    case_ids: list[str] | None = None,
    canonical_key: str = "pressure",
    plot_coord: str = "x",
    normalize_factor: float = 1.0,
    coord_trim: tuple[float | None, float | None] | None = None,
    field_trim: tuple[float | None, float | None] | None = None,
    flip: bool = False,
    **kwargs: Any,
) -> None:
    """GT vs pred line plot along ``plot_coord`` using cell-centered samples of the comparison mesh.

    Resolves VTK array names from ``output.ground_truth_mesh_field_names`` and
    ``output.mesh_field_names`` for ``canonical_key`` (surface defaults). The mesh is reduced to
    cell centers (or points if there are no cells), then passed to ``plot_line`` as a **single**
    polyline-like dataset (values vs sorted ``plot_coord``).

    Extra ``**kwargs`` are forwarded to ``plot_line`` (e.g. ``true_line_kwargs``, ``pred_line_kwargs``,
    ``xlabel``, ``ylabel``, ``title_kwargs``).
    """
    out = Path(output_dir)
    vis_dir = out / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)
    output: OutputConfig = config.output

    field_true, field_pred = _resolve_gt_pred_fields(output, canonical_key)

    ct_raw = coord_trim if coord_trim is not None else (None, None)
    ft_raw = field_trim if field_trim is not None else (None, None)
    ct = (ct_raw[0], ct_raw[1]) if isinstance(ct_raw, (list, tuple)) and len(ct_raw) == 2 else (None, None)
    ft = (ft_raw[0], ft_raw[1]) if isinstance(ft_raw, (list, tuple)) and len(ft_raw) == 2 else (None, None)

    for run_idx, run in enumerate(results):
        if run.get("skipped"):
            continue
        model = run["model"]
        dataset = run["dataset"]
        for row in run.get("per_case") or []:
            cid = row["case_id"]
            if case_ids is not None and cid not in case_ids:
                continue
            mesh = get_comparison_mesh_for_case(row, cid, run_idx, context)
            if mesh is None:
                log_dataset(
                    "benchmark",
                    f"Skip line_plot for {cid!r}: no comparison mesh in context or path",
                )
                continue
            line_mesh = _comparison_mesh_to_line_polydata(mesh)
            fig = plot_line(
                line_mesh,
                plot_coord=plot_coord,
                field_true=field_true,
                field_pred=field_pred,
                normalize_factor=normalize_factor,
                coord_trim=ct,
                field_trim=ft,
                flip=flip,
                **kwargs,
            )
            safe = f"{model}_{dataset}_{cid}_line_{canonical_key}_{plot_coord}.png".replace(
                "/", "_"
            )
            out_png = vis_dir / safe
            fig.savefig(str(out_png), bbox_inches="tight", dpi=150)
            plt.close(fig)
            log_dataset("benchmark", f"Wrote line plot {out_png}")


def register_line_plot() -> None:
    register_visual("line_plot", line_plot)
    register_visual("plot_line", line_plot)  # alias (same implementation)
    register_visual("line_plot_surface", line_plot)  # alias: same as line_plot for surface comparison meshes
    register_visual("line_plot_centerlines", line_plot_centerlines)
