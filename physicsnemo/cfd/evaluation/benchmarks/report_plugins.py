# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Optional PNG report visuals using ``physicsnemo.cfd.bench.visualization``.

Benchmark and inference entrypoints call ``run_optional_report_plugins`` when
``reports.enabled`` and ``reports.visuals`` are set. Optional ``context`` (see
that function) can supply in-memory comparison meshes so plugins avoid reading
large VTU/VTP files from disk.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from physicsnemo.cfd.evaluation.config import Config
from physicsnemo.cfd.evaluation.datasets.progress import log_dataset
from physicsnemo.cfd.evaluation.reports.registry import get_visual, normalize_visuals_config


def _apply_default_case_ids_to_visuals(
    config: Config,
    specs: list[tuple[str, dict[str, Any]]],
) -> list[tuple[str, dict[str, Any]]]:
    """
    Use ``reports.visual_case_ids`` as the default ``case_ids`` for each visual spec.

    Parameters
    ----------
    config : Config
        Root configuration (reads ``reports.visual_case_ids``).
    specs : list of tuple
        Normalized ``(visual_name, kwargs)`` pairs.

    Returns
    -------
    list of tuple
        Specs with ``case_ids`` filled in when absent and a default list exists.
    """
    default = config.reports.visual_case_ids
    if default is None:
        return specs
    out: list[tuple[str, dict[str, Any]]] = []
    for name, kw in specs:
        kw2 = dict(kw)
        if "case_ids" not in kw2:
            kw2["case_ids"] = list(default)
        out.append((name, kw2))
    return out


def run_optional_report_plugins(
    config: Config,
    results: list[dict[str, Any]],
    output_dir: str,
    *,
    context: dict[str, Any] | None = None,
) -> None:
    """
    Run registered report visuals and write ``report_plugins_manifest.json``.

    Parameters
    ----------
    config : Config
        Must have ``reports.enabled`` and a non-empty ``reports.visuals`` list to run plugins.
    results : list of dict
        Benchmark results (same structure as ``run_benchmark`` output).
    output_dir : str
        Directory for the manifest and plot outputs.
    context : dict or None, optional
        Runtime-only data not written to the manifest beyond key names.

        Supported keys:

        - ``comparison_meshes_by_run``: list of dict, same length as ``results``;
          each dict maps ``case_id`` to an in-memory PyVista comparison mesh so
          mesh-based visuals can avoid ``pv.read(comparison_mesh_path)``. If
          ``reports.visual_case_ids`` is set, only those IDs appear per run;
          other cases may still load from disk when comparison meshes were saved.

    Notes
    -----
    Visuals that omit ``case_ids`` receive ``reports.visual_case_ids`` as the
    default filter when that list is set; per-visual ``case_ids`` overrides.
    """
    out_dir = Path(output_dir)
    manifest: dict[str, Any] = {
        "enabled": bool(config.reports.enabled),
        "plugins": config.reports.plugins,
        "save_comparison_meshes": config.reports.save_comparison_meshes,
        "comparison_mesh_subdir": config.reports.comparison_mesh_subdir,
        "visual_case_ids": config.reports.visual_case_ids,
        "context_keys": list(context.keys()) if context else [],
        "visuals_ran": [],
        "visual_errors": [],
    }

    if not config.reports.enabled:
        _write_manifest(out_dir, manifest)
        return

    import physicsnemo.cfd.evaluation.reports  # noqa: F401 — register built-in visuals

    visuals_list = list(config.reports.visuals or [])
    if not visuals_list:
        log_dataset("benchmark", "reports.enabled but no reports.visuals configured; writing manifest only.")
        _write_manifest(out_dir, manifest)
        return

    try:
        specs = normalize_visuals_config(visuals_list)
    except ValueError as e:
        manifest["visual_errors"].append({"stage": "normalize", "error": str(e)})
        _write_manifest(out_dir, manifest)
        raise

    specs = _apply_default_case_ids_to_visuals(config, specs)

    for name, vkwargs in specs:
        try:
            fn = get_visual(name)
            params = inspect.signature(fn).parameters
            if "context" in params:
                fn(config, results, output_dir, context=context, **vkwargs)
            else:
                fn(config, results, output_dir, **vkwargs)
            manifest["visuals_ran"].append({"name": name, "kwargs": vkwargs})
        except Exception as e:
            log_dataset("benchmark", f"Visual {name!r} failed: {e}")
            manifest["visual_errors"].append({"name": name, "error": str(e)})

    _write_manifest(out_dir, manifest)


def _write_manifest(out_dir: Path, manifest: dict[str, Any]) -> None:
    """Persist plugin run metadata to ``report_plugins_manifest.json``."""
    out = out_dir / "report_plugins_manifest.json"
    log_dataset("benchmark", f"Writing report plugin manifest to {out}…")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
