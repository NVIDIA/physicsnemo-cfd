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

"""
Aggregate benchmark results and export tabular reports.

Writes ``benchmark_results.{json,csv,html}`` under the run output directory.
"""

import csv
import html
import json
import math
from pathlib import Path
from typing import Any

from physicsnemo.cfd.evaluation.datasets.progress import log_dataset


def _sanitize_for_strict_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None for RFC 8259–valid JSON."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_strict_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_strict_json(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_for_strict_json(x) for x in obj)
    return obj


def _fmt_html_metric_value(value: Any) -> str:
    """Format a metric cell for HTML; escape textual output."""

    if isinstance(value, bool):
        return html.escape(str(value))
    if isinstance(value, int):
        return f"{value:.6g}"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return html.escape(str(value))
        return f"{value:.6g}"
    return html.escape(str(value))


def write_report(
    results: list[dict[str, Any]],
    output_dir: str | Path,
    formats: list[str] | None = None,
) -> None:
    """
    Write benchmark results in the requested formats.

    Parameters
    ----------
    results : list of dict
        One entry per model×dataset from ``run_benchmark`` (with ``per_case``, ``metrics``, etc.).
    output_dir : str or Path
        Directory for output files.
    formats : list of str or None
        Subset of ``"json"``, ``"csv"``, ``"html"``. Default is all three.

    Raises
    ------
    ValueError
        If an unknown format name is given.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if formats is None:
        formats = ["json", "csv", "html"]
    log_dataset(
        "benchmark",
        f"Writing benchmark reports to {output_dir} (formats: {', '.join(formats)})…",
    )
    for fmt in formats:
        if fmt == "json":
            _write_json(results, output_dir)
        elif fmt == "csv":
            _write_csv(results, output_dir)
        elif fmt == "html":
            _write_html(results, output_dir)
        else:
            raise ValueError(f"Unknown format: {fmt}")


def _write_json(results: list[dict], output_dir: Path) -> None:
    """Write ``benchmark_results.json``."""
    path = output_dir / "benchmark_results.json"
    log_dataset("benchmark", f"Writing {path}…")
    payload = _sanitize_for_strict_json(results)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)


def _write_csv(results: list[dict], output_dir: Path) -> None:
    """Write long-form ``benchmark_results.csv`` (per-case and summary rows)."""
    path = output_dir / "benchmark_results.csv"
    log_dataset("benchmark", f"Writing {path}…")
    rows = []
    for r in results:
        model = r.get("model", "")
        dataset = r.get("dataset", "")
        if r.get("skipped"):
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "case_id": "",
                    "metric": "_skipped",
                    "value": r.get("skip_reason", ""),
                }
            )
            continue
        for case in r.get("per_case", []):
            cid = case.get("case_id", "")
            for mname, value in case.get("metrics", {}).items():
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "case_id": cid,
                        "metric": mname,
                        "value": value,
                    }
                )
        # Also summary row (no case_id)
        for mname, value in r.get("metrics", {}).items():
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "case_id": "",
                    "metric": mname,
                    "value": value,
                }
            )
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["model", "dataset", "case_id", "metric", "value"]
        )
        w.writeheader()
        w.writerows(rows)


def _write_html(results: list[dict], output_dir: Path) -> None:
    """Write a simple HTML summary with tables of metrics and per-case values."""
    path = output_dir / "benchmark_results.html"
    log_dataset("benchmark", f"Writing {path}…")
    lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Benchmark Results</title>",
        "<style>table { border-collapse: collapse; } th, td { border: 1px solid #ccc; padding: 6px 10px; } th { background: #eee; }</style>",
        "</head><body><h1>Benchmark Results</h1>",
    ]
    for r in results:
        model = r.get("model", "")
        dataset = r.get("dataset", "")
        lines.append(
            f"<h2>{html.escape(str(model))} / {html.escape(str(dataset))}</h2>"
        )
        if r.get("skipped"):
            lines.append(
                f"<p><em>Skipped:</em> {html.escape(str(r.get('skip_reason', '')))}</p>"
            )
            continue
        metrics = r.get("metrics", {})
        if metrics:
            lines.append(
                "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>"
            )
            for mname, value in metrics.items():
                lines.append(
                    f"<tr><td>{html.escape(str(mname))}</td><td>"
                    f"{_fmt_html_metric_value(value)}</td></tr>"
                )
            lines.append("</tbody></table>")
        per_case = r.get("per_case", [])
        if per_case:
            lines.append("<h3>Per-case</h3><table><thead><tr><th>Case ID</th>")
            all_metrics = sorted(set(k for c in per_case for k in c.get("metrics", {})))
            for m in all_metrics:
                lines.append(f"<th>{html.escape(str(m))}</th>")
            lines.append("</tr></thead><tbody>")
            for c in per_case:
                line = f"<tr><td>{html.escape(str(c.get('case_id', '')))}</td>"
                for m in all_metrics:
                    val = c.get("metrics", {}).get(m, "")
                    line += f"<td>{_fmt_html_metric_value(val)}</td>"
                line += "</tr>"
                lines.append(line)
            lines.append("</tbody></table>")
    lines.append("</body></html>")
    with open(path, "w") as f:
        f.write("\n".join(lines))
