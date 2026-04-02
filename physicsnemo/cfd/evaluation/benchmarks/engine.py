"""Benchmark runner: config → model x dataset loop → metrics → results.

When ``reports.enabled`` and ``reports.visuals`` are set, optionally writes comparison VTK (if
``reports.save_comparison_meshes``) and runs :func:`report_plugins.run_optional_report_plugins` with
in-memory meshes in ``context["comparison_meshes_by_run"]`` so PNGs avoid disk reads when possible.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from physicsnemo.cfd.evaluation.benchmarks.report import write_report
from physicsnemo.cfd.evaluation.config import (
    Config,
    DatasetConfig,
    ModelConfig,
    OutputConfig,
    ReportsConfig,
    RunConfig,
)
from physicsnemo.cfd.evaluation.datasets import get_adapter
from physicsnemo.cfd.evaluation.datasets.gt_alignment import resolve_dataset_kwargs_for_model
from physicsnemo.cfd.evaluation.datasets.progress import log_dataset
from physicsnemo.cfd.evaluation.inference import get_model_wrapper
from physicsnemo.cfd.evaluation.inference.model_registry import get_inference_domain_for_model
from physicsnemo.cfd.evaluation.metrics import get_metric
from physicsnemo.cfd.evaluation.metrics.mesh_bridge import build_comparison_mesh


def _normalize_metrics_config(metrics: list[str] | list[dict]) -> list[tuple[str, dict]]:
    """Return list of (metric_name, kwargs)."""
    out = []
    for m in metrics:
        if isinstance(m, str):
            out.append((m, {}))
        elif isinstance(m, dict) and "name" in m:
            name = m["name"]
            kwargs = {k: v for k, v in m.items() if k != "name"}
            out.append((name, kwargs))
        else:
            raise ValueError(f"Invalid metric entry: {m}")
    return out


def _effective_inference_domain(model_config: ModelConfig) -> str:
    dom = model_config.inference_domain
    if dom in ("surface", "volume"):
        return dom
    return get_inference_domain_for_model(model_config.name)


def _save_inference_mesh_if_requested(
    *,
    run_config: RunConfig,
    model_config: ModelConfig,
    output_config: OutputConfig,
    wrapper: Any,
    case: Any,
    case_id: str,
    predictions: dict[str, Any],
    output_dir: str,
    dataset_name: str,
) -> None:
    """Write ``inference_<model>_<case>.vtp|vtu`` (predictions only) when ``run.save_inference_mesh``."""
    if not run_config.save_inference_mesh:
        return
    import pyvista as pv

    m_dom = case.inference_domain
    out_path = Path(output_dir) / f"inference_{model_config.name}_{case_id}{'.vtp' if m_dom == 'surface' else '.vtu'}"
    log_dataset(
        dataset_name,
        f"Writing inference mesh (predictions only) to {out_path}…",
    )
    try:
        if m_dom == "surface":
            mesh = pv.read(case.mesh_path)
            if not isinstance(mesh, pv.PolyData):
                mesh = mesh.extract_surface()
            names = output_config.mesh_field_names
        else:
            mesh = pv.read(case.mesh_path)
            if hasattr(mesh, "cast_to_unstructured_grid"):
                mesh = mesh.cast_to_unstructured_grid()
            names = output_config.volume_mesh_field_names

        data_target = mesh.cell_data if wrapper.output_location == "cell" else mesh.point_data
        for canonical_key, mesh_name in names.items():
            if canonical_key in predictions:
                data_target[mesh_name] = predictions[canonical_key]
        mesh.save(str(out_path))
        log_dataset(dataset_name, f"Wrote inference mesh: {out_path}")
    except Exception as ex:
        log_dataset(dataset_name, f"Could not write inference mesh to {out_path}: {ex}")


def _call_metric(
    fn: Any,
    gt: dict,
    predictions: dict,
    *,
    case: Any,
    comparison_mesh: Any,
    metric_dtype: str | None,
    output: OutputConfig,
    mkwargs: dict[str, Any],
) -> Any:
    """Call metric with extended kwargs; fall back for legacy (gt, pred, **mkwargs) only."""
    extended = dict(mkwargs)
    extended.update(
        case=case,
        comparison_mesh=comparison_mesh,
        metric_dtype=metric_dtype,
        output=output,
    )
    try:
        return fn(gt, predictions, **extended)
    except TypeError:
        return fn(gt, predictions, **mkwargs)


def _run_single(
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    metric_names: list[tuple[str, dict]],
    device: str,
    output_dir: str,
    case_ids: list[str] | None,
    output_config: OutputConfig,
    *,
    run_config: RunConfig,
    reports: ReportsConfig | None = None,
    allow_skip_mismatch: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one model on one dataset; return results structure and optional in-memory comparison meshes."""
    adapter_class = get_adapter(dataset_config.name)
    m_dom = _effective_inference_domain(model_config)
    d_dom = adapter_class.inference_domain_from_kwargs(dataset_config.kwargs)
    if m_dom != d_dom:
        reason = (
            f"inference_domain mismatch: model expects {m_dom!r}, "
            f"dataset adapter {dataset_config.name!r} is {d_dom!r}"
        )
        if allow_skip_mismatch:
            print(
                f"[benchmark] SKIP {model_config.name!r} × {dataset_config.name!r}: {reason}",
                flush=True,
            )
            return (
                {
                    "model": model_config.name,
                    "dataset": dataset_config.name,
                    "skipped": True,
                    "skip_reason": reason,
                    "cases": [],
                    "metrics": {},
                    "per_case": [],
                },
                {},
            )
        raise ValueError(reason)

    dkwargs = resolve_dataset_kwargs_for_model(dataset_config.kwargs, model_config.name)
    adapter = adapter_class(root=dataset_config.root, **dkwargs)
    log_dataset(
        dataset_config.name,
        f"Listing cases under root {dataset_config.root!r} (split={dataset_config.split!r})…",
    )
    cases = case_ids if case_ids is not None else adapter.list_cases(split=dataset_config.split)
    if not cases:
        return (
            {
                "model": model_config.name,
                "dataset": dataset_config.name,
                "cases": [],
                "metrics": {},
                "per_case": [],
            },
            {},
        )

    wrapper_class = get_model_wrapper(model_config.name)
    wrapper = wrapper_class()
    wrapper.load(
        checkpoint_path=model_config.checkpoint,
        stats_path=model_config.stats_path,
        device=device,
        **model_config.merged_kwargs_for_load(),
    )

    per_case = []
    all_metric_values: dict[str, list[float]] = {}
    mesh_ctx: dict[str, Any] = {}

    log_dataset(
        dataset_config.name,
        f"Loading {len(cases)} case(s) from root {dataset_config.root!r} "
        f"(model {model_config.name!r})…",
    )
    for cid in cases:
        log_dataset(
            dataset_config.name,
            f"Reading case {cid!r}…",
        )
        case = adapter.load_case(cid)
        model_input = wrapper.prepare_inputs(case)
        raw = wrapper.predict(model_input)
        predictions = wrapper.decode_outputs(raw, case)
        gt = case.ground_truth or {}

        _save_inference_mesh_if_requested(
            run_config=run_config,
            model_config=model_config,
            output_config=output_config,
            wrapper=wrapper,
            case=case,
            case_id=cid,
            predictions=predictions,
            output_dir=output_dir,
            dataset_name=dataset_config.name,
        )

        comparison_mesh = None
        metric_dtype: str | None = None
        try:
            comparison_mesh, metric_dtype = build_comparison_mesh(case, predictions, output_config)
        except Exception as ex:
            log_dataset(
                dataset_config.name,
                f"Warning: comparison mesh not built for case {cid!r}: {ex}",
            )

        case_metrics: dict[str, float] = {}
        for mname, mkwargs in metric_names:
            try:
                fn = get_metric(mname)
                out = _call_metric(
                    fn,
                    gt,
                    predictions,
                    case=case,
                    comparison_mesh=comparison_mesh,
                    metric_dtype=metric_dtype,
                    output=output_config,
                    mkwargs=mkwargs,
                )
                if isinstance(out, dict):
                    for k, v in out.items():
                        key = f"{mname}_{k}" if k else mname
                        case_metrics[key] = float(v)
                        all_metric_values.setdefault(key, []).append(float(v))
                else:
                    case_metrics[mname] = float(out)
                    all_metric_values.setdefault(mname, []).append(float(out))
            except Exception as e:
                log_dataset(dataset_config.name, f"Metric {mname!r} failed for {cid!r}: {e}")
                case_metrics[mname] = float("nan")
                all_metric_values.setdefault(mname, []).append(float("nan"))
        row: dict[str, Any] = {"case_id": cid, "metrics": case_metrics}
        if comparison_mesh is not None and metric_dtype is not None:
            row["metric_dtype"] = metric_dtype
            if reports:
                if reports.save_comparison_meshes:
                    sub = Path(output_dir) / reports.comparison_mesh_subdir
                    sub.mkdir(parents=True, exist_ok=True)
                    ext = ".vtp" if case.inference_domain == "surface" else ".vtu"
                    cmp_p = sub / f"{cid}_comparison{ext}"
                    try:
                        comparison_mesh.save(str(cmp_p))
                        row["comparison_mesh_path"] = str(cmp_p.resolve())
                    except Exception as ex:
                        log_dataset(
                            dataset_config.name,
                            f"Could not save comparison mesh for {cid!r}: {ex}",
                        )
                if reports.enabled and reports.visuals:
                    mesh_ctx[cid] = comparison_mesh
        per_case.append(row)

    # Aggregate (mean over cases)
    metrics_summary = {}
    for mname, values in all_metric_values.items():
        valid = [v for v in values if v == v]  # filter nan
        metrics_summary[mname] = sum(valid) / len(valid) if valid else float("nan")

    return (
        {
            "model": model_config.name,
            "dataset": dataset_config.name,
            "cases": cases,
            "metrics": metrics_summary,
            "per_case": per_case,
        },
        mesh_ctx,
    )


def _case_ids_for_run(
    dataset_case_ids: list[str] | None,
    case_id_override: str | None,
) -> list[str] | None:
    """If ``case_id_override`` is set, run only that case; else use dataset config."""
    if case_id_override:
        return [case_id_override]
    return dataset_case_ids


def run_benchmark(
    config: Config,
    *,
    case_id: str | None = None,
) -> list[dict[str, Any]]:
    """Run benchmark from config (single or matrix mode). Returns list of run results.

    If ``case_id`` is set, only that case is evaluated (overrides ``dataset.case_ids``).
    """
    import physicsnemo.cfd.evaluation.datasets.adapters  # noqa: F401
    import physicsnemo.cfd.evaluation.inference.wrappers  # noqa: F401
    import physicsnemo.cfd.evaluation.metrics  # noqa: F401 — registers built-in metrics

    metric_specs = _normalize_metrics_config(config.metrics)
    device = config.run.device
    output_dir = config.run.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if config.benchmark.reproducibility.log_env:
        env_log = Path(output_dir) / "env.json"
        log_dataset("benchmark", f"Writing environment log to {env_log}…")
        with open(env_log, "w") as f:
            json.dump(dict(os.environ), f, indent=2)

    results: list[dict[str, Any]] = []
    meshes_by_run: list[dict[str, Any]] = []

    if config.benchmark.mode == "single":
        case_ids = _case_ids_for_run(config.dataset.case_ids, case_id)
        res, mesh_ctx = _run_single(
            config.model,
            config.dataset,
            metric_specs,
            device,
            output_dir,
            case_ids,
            config.output,
            run_config=config.run,
            reports=config.reports,
            allow_skip_mismatch=False,
        )
        results.append(res)
        meshes_by_run.append(mesh_ctx)
    else:
        models = config.benchmark.models or [config.model]
        datasets = config.benchmark.datasets or [config.dataset]
        for m_cfg in models:
            for d_cfg in datasets:
                res, mesh_ctx = _run_single(
                    m_cfg,
                    d_cfg,
                    metric_specs,
                    device,
                    output_dir,
                    _case_ids_for_run(d_cfg.case_ids, case_id),
                    config.output,
                    run_config=config.run,
                    reports=config.reports,
                    allow_skip_mismatch=True,
                )
                results.append(res)
                meshes_by_run.append(mesh_ctx)

    if config.benchmark.reproducibility.save_artifacts:
        artifacts = Path(output_dir) / "benchmark_artifacts.json"
        skipped = [r for r in results if r.get("skipped")]
        log_dataset("benchmark", f"Writing artifacts to {artifacts}…")
        with open(artifacts, "w") as f:
            json.dump(
                {
                    "config": _config_to_dict(config),
                    "results_summary": [
                        {
                            "model": r["model"],
                            "dataset": r["dataset"],
                            "metrics": r["metrics"],
                            "skipped": r.get("skipped", False),
                            "skip_reason": r.get("skip_reason"),
                        }
                        for r in results
                    ],
                    "skipped_runs": skipped,
                },
                f,
                indent=2,
            )

    write_report(results, output_dir, formats=["json", "csv", "html"])

    if config.reports.enabled and config.reports.visuals:
        import physicsnemo.cfd.evaluation.reports  # noqa: F401 — register built-in visuals

        from physicsnemo.cfd.evaluation.benchmarks.report_plugins import run_optional_report_plugins

        log_dataset("benchmark", "Running reports.visuals from benchmark results…")
        run_optional_report_plugins(
            config,
            results,
            output_dir,
            context={"comparison_meshes_by_run": meshes_by_run},
        )

    return results


def _config_to_dict(c: Config) -> dict:
    """Serializable config for artifacts."""
    return {
        "run": {
            "device": c.run.device,
            "output_dir": c.run.output_dir,
            "seed": c.run.seed,
            "batch_size": c.run.batch_size,
            "save_inference_mesh": c.run.save_inference_mesh,
        },
        "model": {
            "name": c.model.name,
            "checkpoint": c.model.checkpoint,
            "stats_path": c.model.stats_path,
            "kwargs": c.model.kwargs,
            "inference_domain": c.model.inference_domain,
        },
        "dataset": {
            "name": c.dataset.name,
            "root": c.dataset.root,
            "split": c.dataset.split,
            "case_ids": c.dataset.case_ids,
            "kwargs": c.dataset.kwargs,
        },
        "output": {
            "mesh_field_names": c.output.mesh_field_names,
            "volume_mesh_field_names": c.output.volume_mesh_field_names,
            "ground_truth_mesh_field_names": c.output.ground_truth_mesh_field_names,
            "ground_truth_volume_mesh_field_names": c.output.ground_truth_volume_mesh_field_names,
            "streamlines_vector_canonical": c.output.streamlines_vector_canonical,
        },
        "metrics": c.metrics,
        "reports": {
            "enabled": c.reports.enabled,
            "plugins": c.reports.plugins,
            "save_comparison_meshes": c.reports.save_comparison_meshes,
            "comparison_mesh_subdir": c.reports.comparison_mesh_subdir,
            "visuals": c.reports.visuals,
        },
        "benchmark": {
            "mode": c.benchmark.mode,
            "reproducibility": {
                "log_env": c.benchmark.reproducibility.log_env,
                "save_artifacts": c.benchmark.reproducibility.save_artifacts,
            },
        },
    }
