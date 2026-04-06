# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Design scatter / trend visuals wrapping ``physicsnemo.cfd.postprocessing_tools.visualization.utils``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from physicsnemo.cfd.postprocessing_tools.visualization.utils import plot_design_scatter, plot_design_trend
from physicsnemo.cfd.evaluation.config import Config
from physicsnemo.cfd.evaluation.datasets.progress import log_dataset
from physicsnemo.cfd.evaluation.reports.registry import register_visual


def _design_skip_hint(per_case: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> str:
    """Explain why design plots may find no finite (true, pred) pairs."""
    if not per_case:
        return ""
    m = (per_case[0].get("metrics") or {}) if per_case else {}
    parts: list[str] = []
    for p in pairs:
        tk = str(p.get("true_key", ""))
        pk = str(p.get("pred_key", ""))
        if tk == "drag_error_true" and pk == "drag_error_pred":
            if "drag_error" in m and tk not in m:
                parts.append(
                    "metrics only have scalar drag_error (missing drag_error_true / drag_error_pred). "
                    "Re-run the benchmark after updating drag_error to return Cd true/pred dict entries, "
                    "and ensure `python` imports that code (e.g. `pip install -e .` from the repo root)."
                )
            elif tk in m and pk in m and (m[tk] != m[tk] or m[pk] != m[pk]):
                parts.append(f"{tk!r} or {pk!r} is NaN (finite values required).")
        if tk == "lift_error_true" and pk == "lift_error_pred":
            if "lift_error" in m and tk not in m:
                parts.append(
                    "metrics only have scalar lift_error (missing lift_error_true / lift_error_pred); "
                    "re-run with updated lift_error metric."
                )
            elif tk in m and pk in m and (m[tk] != m[tk] or m[pk] != m[pk]):
                parts.append(f"{tk!r} or {pk!r} is NaN (finite values required).")
    if not parts:
        return ""
    return " " + " ".join(dict.fromkeys(parts))


def design_scatter(
    config: Config,
    results: list[dict[str, Any]],
    output_dir: str,
    *,
    context: dict[str, Any] | None = None,
    pairs: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> None:
    """Scatter of pred vs true per run, using ``per_case[].metrics`` keys from *pairs*.

    Each entry in *pairs* is ``{name, true_key, pred_key}`` (metric names in ``metrics``).
    """
    del context, config
    if not pairs:
        raise ValueError("design_scatter requires non-empty ``pairs`` (list of name/true_key/pred_key).")
    out = Path(output_dir) / "visuals"
    out.mkdir(parents=True, exist_ok=True)
    for run in results:
        if run.get("skipped"):
            continue
        model = run["model"]
        dataset = run["dataset"]
        true_data_dict: dict[str, list[float]] = {}
        pred_data_dict: dict[str, list[float]] = {}
        for p in pairs:
            name = str(p["name"])
            tk = str(p["true_key"])
            pk = str(p["pred_key"])
            true_data_dict[name] = []
            pred_data_dict[name] = []
            for row in run.get("per_case") or []:
                m = row.get("metrics") or {}
                if tk in m and pk in m and m[tk] == m[tk] and m[pk] == m[pk]:
                    true_data_dict[name].append(float(m[tk]))
                    pred_data_dict[name].append(float(m[pk]))
        if not any(len(true_data_dict[k]) > 0 for k in true_data_dict):
            hint = _design_skip_hint(run.get("per_case") or [], pairs or [])
            log_dataset(
                "benchmark",
                f"Skip design_scatter for {model}/{dataset}: no finite true/pred points for pairs "
                f"{pairs!r}.{hint}",
            )
            continue
        fig, _ = plot_design_scatter(true_data_dict, pred_data_dict, **kwargs)
        safe = f"{model}_{dataset}_design_scatter.png".replace("/", "_")
        fig.savefig(str(out / safe), bbox_inches="tight", dpi=150)
        plt.close(fig)
        log_dataset("benchmark", f"Wrote design scatter {out / safe}")


def design_trend(
    config: Config,
    results: list[dict[str, Any]],
    output_dir: str,
    *,
    context: dict[str, Any] | None = None,
    pairs: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> None:
    """Trend plot; optional ``idx_key`` per pair selects a metric for x labels (else ``case_id``)."""
    del context, config
    if not pairs:
        raise ValueError("design_trend requires non-empty ``pairs``.")
    out = Path(output_dir) / "visuals"
    out.mkdir(parents=True, exist_ok=True)
    for run in results:
        if run.get("skipped"):
            continue
        model = run["model"]
        dataset = run["dataset"]
        true_data_dict: dict[str, list[float]] = {}
        pred_data_dict: dict[str, list[float]] = {}
        idx_dict: dict[str, list[Any]] = {}
        for p in pairs:
            name = str(p["name"])
            tk = str(p["true_key"])
            pk = str(p["pred_key"])
            idx_key = p.get("idx_key")
            true_data_dict[name] = []
            pred_data_dict[name] = []
            idx_dict[name] = []
            for row in run.get("per_case") or []:
                m = row.get("metrics") or {}
                if tk not in m or pk not in m:
                    continue
                if m[tk] != m[tk] or m[pk] != m[pk]:
                    continue
                true_data_dict[name].append(float(m[tk]))
                pred_data_dict[name].append(float(m[pk]))
                if idx_key:
                    idx_dict[name].append(m.get(idx_key, row.get("case_id")))
                else:
                    idx_dict[name].append(str(row.get("case_id", "")))
        if not any(len(true_data_dict[k]) > 0 for k in true_data_dict):
            hint = _design_skip_hint(run.get("per_case") or [], pairs or [])
            log_dataset(
                "benchmark",
                f"Skip design_trend for {model}/{dataset}: no finite true/pred points for pairs "
                f"{pairs!r}.{hint}",
            )
            continue
        res = plot_design_trend(true_data_dict, pred_data_dict, idx_dict, **kwargs)
        fig = res[0]
        safe = f"{model}_{dataset}_design_trend.png".replace("/", "_")
        fig.savefig(str(out / safe), bbox_inches="tight", dpi=150)
        plt.close(fig)
        log_dataset("benchmark", f"Wrote design trend {out / safe}")


def register_design_visuals() -> None:
    register_visual("design_scatter", design_scatter)
    register_visual("design_trend", design_trend)
