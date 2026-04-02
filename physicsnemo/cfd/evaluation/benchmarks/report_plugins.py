# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional visuals (plots via ``physicsnemo.cfd.bench.visualization``).

The benchmark engine does **not** call this module: benchmarking stays scalar-only (JSON/CSV/HTML)
with in-memory comparison meshes for metrics only.

The **inference** CLI (``physicsnemo.cfd.evaluation.inference.run``) calls
:func:`run_optional_report_plugins` after saving the prediction mesh when ``reports.enabled`` and
``reports.visuals`` are set: it builds a comparison mesh, writes ``{case_id}_comparison.vtp|vtu`` when
needed, then runs registered visuals. You can also call :func:`run_optional_report_plugins` from
your own script when you have ``results`` and want a manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from physicsnemo.cfd.evaluation.config import Config
from physicsnemo.cfd.evaluation.datasets.progress import log_dataset
from physicsnemo.cfd.evaluation.reports.registry import get_visual, normalize_visuals_config


def run_optional_report_plugins(
    config: Config,
    results: list[dict[str, Any]],
    output_dir: str,
) -> None:
    """Run registered visuals when ``config.reports.enabled``; write manifest and optional legacy plugins note."""
    out_dir = Path(output_dir)
    manifest: dict[str, Any] = {
        "enabled": bool(config.reports.enabled),
        "plugins": config.reports.plugins,
        "save_comparison_meshes": config.reports.save_comparison_meshes,
        "comparison_mesh_subdir": config.reports.comparison_mesh_subdir,
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

    for name, vkwargs in specs:
        try:
            fn = get_visual(name)
            fn(config, results, output_dir, **vkwargs)
            manifest["visuals_ran"].append({"name": name, "kwargs": vkwargs})
        except Exception as e:
            log_dataset("benchmark", f"Visual {name!r} failed: {e}")
            manifest["visual_errors"].append({"name": name, "error": str(e)})

    _write_manifest(out_dir, manifest)


def _write_manifest(out_dir: Path, manifest: dict[str, Any]) -> None:
    out = out_dir / "report_plugins_manifest.json"
    log_dataset("benchmark", f"Writing report plugin manifest to {out}…")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
