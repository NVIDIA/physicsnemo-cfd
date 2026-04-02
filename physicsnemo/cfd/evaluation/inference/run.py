# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for single-case inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from physicsnemo.cfd.evaluation.benchmarks.report_plugins import run_optional_report_plugins
from physicsnemo.cfd.evaluation.config import Config, load_config
import physicsnemo.cfd.evaluation.datasets.adapters  # noqa: F401
from physicsnemo.cfd.evaluation.datasets import get_adapter
from physicsnemo.cfd.evaluation.datasets.gt_alignment import resolve_dataset_kwargs_for_model
from physicsnemo.cfd.evaluation.datasets.progress import log_dataset
from physicsnemo.cfd.evaluation.inference import get_model_wrapper
from physicsnemo.cfd.evaluation.inference.model_registry import get_inference_domain_for_model
from physicsnemo.cfd.evaluation.inference.progress import log_inference
from physicsnemo.cfd.evaluation.metrics.mesh_bridge import build_comparison_mesh


def _effective_model_inference_domain(model_name: str, override: str | None) -> str:
    if override in ("surface", "volume"):
        return override
    return get_inference_domain_for_model(model_name)


def _parse_overrides(args: list[str]) -> dict[str, str]:
    overrides = {}
    for a in args:
        if "=" in a:
            s = a[2:] if a.startswith("--") else a
            key, _, val = s.partition("=")
            overrides[key.strip()] = val.strip()
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for one model on one case.")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    parser.add_argument(
        "--base-config",
        default=None,
        help="Optional YAML/JSON merged before --config (shared defaults).",
    )
    parser.add_argument("--case-id", default=None, help="Case ID (default: first case from dataset)")
    parser.add_argument("overrides", nargs="*", help="Key=value overrides, e.g. run.device=cuda:1")
    args = parser.parse_args()
    overrides = _parse_overrides(getattr(args, "overrides", []))
    config = load_config(args.config, overrides, base=args.base_config)

    adapter_class = get_adapter(config.dataset.name)
    dkwargs = resolve_dataset_kwargs_for_model(config.dataset.kwargs, config.model.name)
    adapter = adapter_class(root=config.dataset.root, **dkwargs)
    log_dataset(
        config.dataset.name,
        f"Listing cases under root {config.dataset.root!r} (split={config.dataset.split!r})…",
    )
    case_ids = config.dataset.case_ids or adapter.list_cases(split=config.dataset.split)
    if not case_ids:
        raise SystemExit("No cases found for dataset.")
    case_id = args.case_id if args.case_id else case_ids[0]
    log_dataset(
        config.dataset.name,
        f"Reading case {case_id!r} from dataset root {config.dataset.root!r}…",
    )
    case = adapter.load_case(case_id)

    model_domain = _effective_model_inference_domain(
        config.model.name, config.model.inference_domain
    )
    adapter_domain = adapter_class.inference_domain_from_kwargs(dkwargs)
    if model_domain != adapter_domain:
        raise SystemExit(
            f"Inference domain mismatch: model {config.model.name!r} expects {model_domain!r}, "
            f"dataset adapter {config.dataset.name!r} provides {adapter_domain!r}."
        )

    wrapper_class = get_model_wrapper(config.model.name)
    wrapper = wrapper_class()
    wrapper.load(
        checkpoint_path=config.model.checkpoint,
        stats_path=config.model.stats_path,
        device=config.run.device,
        **config.model.merged_kwargs_for_load(),
    )
    model_input = wrapper.prepare_inputs(case)
    raw = wrapper.predict(model_input)
    predictions = wrapper.decode_outputs(raw, case)

    out_dir = Path(config.run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.run.save_inference_mesh:
        import pyvista as pv

        out_path = out_dir / f"inference_{config.model.name}_{case_id}{'.vtp' if model_domain == 'surface' else '.vtu'}"
        log_inference(
            config.model.name,
            f"Saving predictions to file ({model_domain}): {out_path}",
        )
        log_inference(
            config.model.name,
            f"Reading mesh file to attach fields: {case.mesh_path}",
        )

        if model_domain == "surface":
            mesh = pv.read(case.mesh_path)
            if not isinstance(mesh, pv.PolyData):
                mesh = mesh.extract_surface()
            names = config.output.mesh_field_names
        else:
            mesh = pv.read(case.mesh_path)
            if hasattr(mesh, "cast_to_unstructured_grid"):
                mesh = mesh.cast_to_unstructured_grid()
            names = config.output.volume_mesh_field_names

        data_target = mesh.cell_data if wrapper.output_location == "cell" else mesh.point_data
        for canonical_key, mesh_name in names.items():
            if canonical_key in predictions:
                data_target[mesh_name] = predictions[canonical_key]
        mesh.save(str(out_path))
        log_inference(config.model.name, f"Wrote mesh with prediction fields: {out_path}")
    else:
        log_inference(
            config.model.name,
            f"Skipping inference mesh file (run.save_inference_mesh: false); "
            f"would be inference_{config.model.name}_{case_id}.vtp|vtu.",
        )

    _maybe_save_comparison_mesh_and_run_visuals(
        config=config,
        case_id=case_id,
        case=case,
        predictions=predictions,
        out_dir=out_dir,
    )


def _maybe_save_comparison_mesh_and_run_visuals(
    *,
    config: Config,
    case_id: str,
    case: Any,
    predictions: dict[str, Any],
    out_dir: Path,
) -> None:
    """Build GT+pred comparison mesh, optionally save VTK, run ``reports.visuals`` when enabled."""
    rep = config.reports
    want_save = rep.save_comparison_meshes or (
        rep.enabled and bool(rep.visuals)
    )
    if not want_save:
        return

    try:
        comparison_mesh, metric_dtype = build_comparison_mesh(
            case, predictions, config.output
        )
    except Exception as ex:
        log_inference(
            config.model.name,
            f"Skipping comparison mesh / visuals: build_comparison_mesh failed: {ex}",
        )
        return

    sub = out_dir / rep.comparison_mesh_subdir
    sub.mkdir(parents=True, exist_ok=True)
    ext = ".vtp" if case.inference_domain == "surface" else ".vtu"
    cmp_path = sub / f"{case_id}_comparison{ext}"
    try:
        comparison_mesh.save(str(cmp_path))
    except Exception as ex:
        log_inference(
            config.model.name,
            f"Could not save comparison mesh to {cmp_path}: {ex}",
        )
        return

    log_inference(
        config.model.name,
        f"Wrote comparison mesh for visuals/metrics bridge: {cmp_path}",
    )

    if not rep.enabled or not rep.visuals:
        return

    import physicsnemo.cfd.evaluation.reports  # noqa: F401 — register built-in visuals

    results: list[dict[str, Any]] = [
        {
            "model": config.model.name,
            "dataset": config.dataset.name,
            "skipped": False,
            "per_case": [
                {
                    "case_id": case_id,
                    "metric_dtype": metric_dtype,
                    "comparison_mesh_path": str(cmp_path.resolve()),
                    "metrics": {},
                }
            ],
        }
    ]
    log_inference(
        config.model.name,
        f"Running reports.visuals ({len(rep.visuals)}) under {out_dir}…",
    )
    run_optional_report_plugins(config, results, str(out_dir))


if __name__ == "__main__":
    main()
