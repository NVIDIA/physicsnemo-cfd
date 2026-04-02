# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility CLI: forwards to the benchmark evaluation engine (metrics, tables, visuals)."""

from __future__ import annotations

import argparse
import sys

from physicsnemo.cfd.evaluation.benchmarks.engine import run_benchmark
from physicsnemo.cfd.evaluation.config import Config, load_config
import physicsnemo.cfd.evaluation.datasets.adapters  # noqa: F401
from physicsnemo.cfd.evaluation.datasets import get_adapter
from physicsnemo.cfd.evaluation.datasets.gt_alignment import resolve_dataset_kwargs_for_model


def _parse_overrides(args: list[str]) -> dict[str, str]:
    overrides = {}
    for a in args:
        if "=" in a:
            s = a[2:] if a.startswith("--") else a
            key, _, val = s.partition("=")
            overrides[key.strip()] = val.strip()
    return overrides


def _first_case_id(config: Config) -> str | None:
    """Legacy inference default: first case when ``--case-id`` is omitted."""
    adapter_class = get_adapter(config.dataset.name)
    dkwargs = resolve_dataset_kwargs_for_model(config.dataset.kwargs, config.model.name)
    adapter = adapter_class(root=config.dataset.root, **dkwargs)
    case_ids = config.dataset.case_ids or adapter.list_cases(split=config.dataset.split)
    if not case_ids:
        return None
    return case_ids[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper for the benchmark evaluation engine. Prefer: "
            "python -m physicsnemo.cfd.evaluation.benchmarks.run --config evaluation_config.yaml"
        ),
    )
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    parser.add_argument(
        "--base-config",
        default=None,
        help="Optional YAML/JSON merged before --config (shared defaults).",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Case ID (default: first case from dataset, matching legacy single-case inference).",
    )
    parser.add_argument("overrides", nargs="*", help="Key=value overrides, e.g. run.device=cuda:1")
    args = parser.parse_args()
    overrides = _parse_overrides(getattr(args, "overrides", []))
    config = load_config(args.config, overrides, base=args.base_config)

    case_id = args.case_id if args.case_id else _first_case_id(config)
    if case_id is None:
        raise SystemExit("No cases found for dataset.")

    print(
        "[evaluation] physicsnemo.cfd.evaluation.inference forwards to the benchmark engine; "
        "prefer: python -m physicsnemo.cfd.evaluation.benchmarks.run",
        file=sys.stderr,
    )
    results = run_benchmark(config, case_id=case_id)
    print(f"Completed {len(results)} run(s). Results in {config.run.output_dir}")


if __name__ == "__main__":
    main()
