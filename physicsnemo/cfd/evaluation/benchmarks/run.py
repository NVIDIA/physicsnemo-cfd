"""CLI entrypoint for benchmark runs."""

import argparse

from physicsnemo.cfd.evaluation.benchmarks.engine import run_benchmark
from physicsnemo.cfd.evaluation.config import load_config


def _parse_overrides(args: list[str]) -> dict[str, str]:
    overrides = {}
    for a in args:
        if "=" in a:
            s = a[2:] if a.startswith("--") else a
            key, _, val = s.partition("=")
            overrides[key.strip()] = val.strip()
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark from config.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML/JSON config (merge on top of --base-config if set).",
    )
    parser.add_argument(
        "--base-config",
        default=None,
        help="Optional YAML merged first (e.g. inference_config.yaml); --config overlays it.",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Run only this case ID (overrides dataset.case_ids in config).",
    )
    parser.add_argument("overrides", nargs="*", help="Key=value overrides, e.g. run.device=cuda:1")
    args = parser.parse_args()
    overrides = _parse_overrides(getattr(args, "overrides", []))
    config = load_config(args.config, overrides, base=args.base_config)
    results = run_benchmark(config, case_id=args.case_id)
    print(f"Completed {len(results)} run(s). Results in {config.run.output_dir}")


if __name__ == "__main__":
    main()
