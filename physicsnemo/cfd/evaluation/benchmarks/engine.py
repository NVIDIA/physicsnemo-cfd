"""Benchmark runner: config → model x dataset loop → metrics → results.

Comparison meshes are built in memory for metrics only; the benchmark path does not write
VTP/VTU files or run visualization plugins (see ``report_plugins.run_optional_report_plugins`` for
optional post-processing outside this CLI).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from physicsnemo.cfd.evaluation.benchmarks.report import write_report
from physicsnemo.cfd.evaluation.config import Config, ModelConfig, DatasetConfig, OutputConfig
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
    allow_skip_mismatch: bool = False,
) -> dict[str, Any]:
    """Run one model on one dataset; return results structure."""
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
            return {
                "model": model_config.name,
                "dataset": dataset_config.name,
                "skipped": True,
                "skip_reason": reason,
                "cases": [],
                "metrics": {},
                "per_case": [],
            }
        raise ValueError(reason)

    dkwargs = resolve_dataset_kwargs_for_model(dataset_config.kwargs, model_config.name)
    adapter = adapter_class(root=dataset_config.root, **dkwargs)
    log_dataset(
        dataset_config.name,
        f"Listing cases under root {dataset_config.root!r} (split={dataset_config.split!r})…",
    )
    cases = case_ids if case_ids is not None else adapter.list_cases(split=dataset_config.split)
    if not cases:
        return {
            "model": model_config.name,
            "dataset": dataset_config.name,
            "cases": [],
            "metrics": {},
            "per_case": [],
        }

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
        per_case.append(row)

    # Aggregate (mean over cases)
    metrics_summary = {}
    for mname, values in all_metric_values.items():
        valid = [v for v in values if v == v]  # filter nan
        metrics_summary[mname] = sum(valid) / len(valid) if valid else float("nan")

    return {
        "model": model_config.name,
        "dataset": dataset_config.name,
        "cases": cases,
        "metrics": metrics_summary,
        "per_case": per_case,
    }


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

    if config.benchmark.mode == "single":
        case_ids = _case_ids_for_run(config.dataset.case_ids, case_id)
        res = _run_single(
            config.model,
            config.dataset,
            metric_specs,
            device,
            output_dir,
            case_ids,
            config.output,
            allow_skip_mismatch=False,
        )
        results.append(res)
    else:
        models = config.benchmark.models or [config.model]
        datasets = config.benchmark.datasets or [config.dataset]
        for m_cfg in models:
            for d_cfg in datasets:
                res = _run_single(
                    m_cfg,
                    d_cfg,
                    metric_specs,
                    device,
                    output_dir,
                    _case_ids_for_run(d_cfg.case_ids, case_id),
                    config.output,
                    allow_skip_mismatch=True,
                )
                results.append(res)

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
