# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional post-scalar report plugins (streamlines, plots via bench.visualization)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from physicsnemo.cfd.evaluation.config import Config
from physicsnemo.cfd.evaluation.datasets.progress import log_dataset


def run_optional_report_plugins(
    config: Config,
    results: list[dict[str, Any]],
    output_dir: str,
) -> None:
    """If ``config.reports.enabled``, run configured plugins and write a small manifest."""
    if not config.reports.enabled:
        return
    out = Path(output_dir) / "report_plugins_manifest.json"
    manifest = {
        "enabled": True,
        "plugins": config.reports.plugins,
        "note": (
            "Plot/streamline plugins are optional; wire bench.visualization helpers here "
            "when you add concrete plugin names and inputs."
        ),
        "runs": len(results),
    }
    log_dataset("benchmark", f"Writing report plugin manifest to {out}…")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
