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

"""Sample-level sparsification / AUSE visual for UQ benchmark runs.

Sparsification answers the active-learning question at the *geometry* level: rank the
validation geometries by predicted uncertainty, drop the most-uncertain first, and check that
the error of what remains falls toward the **oracle** (which drops the true-worst-error first).
The gap between the two curves is the **AUSE** (Area Under the Sparsification Error curve; lower
is better, 0 == oracle). The flat **full-RMSE** line is the do-nothing / random baseline.

This visual consumes the ``uq_sparsification`` payload the benchmark engine attaches to each
result (built by
:func:`~physicsnemo.cfd.evaluation.benchmarks.uq_inference.compute_sparsification_payload` from
the sample metrics ``sparsification_ause`` / ``sparsification_ause_epistemic`` and, for the
decision-relevant drag panel, ``drag_uq``). One figure is written per dataset with a panel per
(metric, series) and every model overlaid, so GP and MC-Dropout uncertainties can be compared
against each other and the oracle. Deterministic models produce no payload and are skipped.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physicsnemo.cfd.evaluation.datasets.progress import log_dataset
from physicsnemo.cfd.evaluation.reports.registry import register_visual
from physicsnemo.cfd.evaluation.reports.visual_filenames import benchmark_visual_png

_PALETTE = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
)


def _panel_title(metric_name: str, series: str) -> str:
    """Human-readable panel title from a ``(metric, series)`` key."""
    label = metric_name
    if label.startswith("sparsification_ause"):
        kind = "epistemic std" if label.endswith("_epistemic") else "total std"
        return f"{series} \u2014 rank by {kind}"
    if label == "drag_uq":
        return f"Drag (Cd) \u2014 rank by {series} std"
    return f"{metric_name}:{series}"


def _collect_panels(runs: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[tuple[str, str]]:
    """Ordered, de-duplicated ``(metric_name, series)`` panel keys across all (run, payload) pairs."""
    seen: dict[tuple[str, str], None] = {}
    for _, payload in runs:
        for metric_name in sorted(payload):
            for series in payload[metric_name]:
                seen.setdefault((metric_name, series), None)
    return list(seen)


def _draw_panel(
    ax: Any,
    panel: tuple[str, str],
    runs: list[tuple[dict[str, Any], dict[str, Any]]],
    color_by_model: dict[str, str],
) -> bool:
    """Overlay every model's curves for one panel; return ``True`` if anything was drawn."""
    metric_name, series = panel
    drawn = False
    for run, payload in runs:
        curve = (payload.get(metric_name) or {}).get(series)
        if not curve:
            continue
        model = run["model"]
        color = color_by_model[model]
        fr = curve["fractions"]
        ax.plot(
            fr,
            curve["by_uncertainty"],
            color=color,
            lw=2.2,
            label=f"{model} (AUSE={curve['ause']:.3f})",
        )
        ax.plot(fr, curve["oracle"], color=color, ls="--", lw=1.5, alpha=0.8)
        ax.axhline(curve["full"], color=color, ls=":", lw=1.1, alpha=0.6)
        drawn = True
    if drawn:
        ax.set_title(_panel_title(metric_name, series), fontsize=11)
        ax.set_xlabel("Fraction of most-uncertain geometries removed")
        ax.set_ylabel("RMSE of retained geometries")
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    return drawn


def sparsification_plot(
    config: Any,
    results: list[dict[str, Any]],
    output_dir: str,
    *,
    context: dict[str, Any] | None = None,
    max_cols: int = 3,
    dpi: int = 140,
    **_: Any,
) -> None:
    """Write one sparsification figure per dataset (panels = metric/series, models overlaid)."""
    del config
    # Curve payloads are passed in ``context`` aligned with ``results`` by index (kept off the
    # result dicts so the numpy arrays never reach the JSON report). Fall back to a per-run key for
    # back-compat / direct callers.
    payloads = (context or {}).get("uq_sparsification_by_run") or []
    paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for i, run in enumerate(results):
        if run.get("skipped"):
            continue
        payload = payloads[i] if i < len(payloads) else None
        if not payload:
            payload = run.get("uq_sparsification")
        if payload:
            paired.append((run, payload))
    if not paired:
        log_dataset(
            "benchmark",
            "Skip sparsification_plot: no run carries a uq_sparsification payload "
            "(need a probabilistic model and the sparsification_ause / drag_uq metrics).",
        )
        return

    out = Path(output_dir) / "visuals"
    out.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for run, payload in paired:
        datasets.setdefault(run["dataset"], []).append((run, payload))

    for dataset, runs in datasets.items():
        panels = _collect_panels(runs)
        if not panels:
            continue
        models = sorted({run["model"] for run, _ in runs})
        color_by_model = {
            m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(models)
        }
        n = len(panels)
        ncols = min(max_cols, n)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6.2 * ncols, 4.8 * nrows), squeeze=False
        )
        flat = [ax for row in axes for ax in row]
        any_drawn = False
        for ax, panel in zip(flat, panels):
            any_drawn |= _draw_panel(ax, panel, runs, color_by_model)
        for ax in flat[n:]:
            ax.axis("off")
        if not any_drawn:
            plt.close(fig)
            continue
        fig.suptitle(
            f"Sample-level sparsification \u2014 {dataset} "
            "(solid: by uncertainty, dashed: oracle, dotted: full RMSE)",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        safe = benchmark_visual_png("sparsification", dataset)
        fig.savefig(str(out / safe), bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        log_dataset("benchmark", f"Wrote sparsification plot {out / safe}")


def register_sparsification_visual() -> None:
    """Register the ``sparsification_plot`` visual."""
    register_visual("sparsification_plot", sparsification_plot)
