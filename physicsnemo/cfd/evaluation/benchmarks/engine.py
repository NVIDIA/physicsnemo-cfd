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
Benchmark evaluation driver for config-driven model-by-dataset runs.

Loads registered dataset adapters and model wrappers, evaluates configured
metrics per case, and aggregates per-metric means. Can write comparison VTK,
tabular reports (JSON/CSV/HTML), and PNG visuals from the report plugin
pipeline.

When ``reports.enabled`` and ``reports.visuals`` are set, comparison meshes may
be written under ``reports.comparison_mesh_subdir`` and/or kept in memory for
plugins. Use ``reports.visual_case_ids`` to limit which cases retain meshes in
memory; other cases may still load from ``comparison_mesh_path`` on disk if
meshes were saved.

When ``run.metrics_cache`` is enabled, a valid cache entry skips per-case VTK
load and inference for that case. The cache stores scalars only and does not
replace mesh or visualization workflows.

Multi-GPU: launch with ``torchrun`` (or any launcher that sets ``WORLD_SIZE`` /
``LOCAL_RANK``) so ``physicsnemo.distributed.DistributedManager`` initializes.
With ``run.distributed`` true (default) and world size > 1, cases are strided
across ranks (``cases[rank::world_size]``), results are merged on rank 0, then
broadcast to all ranks; JSON/CSV/HTML, artifacts, and report plugins run on
rank 0 only. Inference uses ``str(dm.device)`` per rank when ``DistributedManager`` is active.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from physicsnemo.cfd.evaluation.benchmarks.distributed_utils import (
    effective_device_str,
    gather_merge_benchmark_outputs,
    log_distributed_context,
    shard_tuple,
    try_get_distributed_manager,
)

from physicsnemo.cfd.evaluation.benchmarks.metrics_cache import (
    metrics_cache_file_path,
    metrics_cache_fingerprint,
    metrics_from_cache_json,
    output_config_to_fingerprint_dict,
    read_metrics_cache,
    resolve_metrics_cache_root,
    write_metrics_cache,
)
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


def _retain_comparison_mesh_for_visual_context(reports: ReportsConfig | None, case_id: str) -> bool:
    """
    Determine if the comparison mesh for this case should be kept for report visuals.

    Parameters
    ----------
    reports : ReportsConfig or None
        Report configuration (must be enabled with visuals for retention).
    case_id : str
        Case identifier.

    Returns
    -------
    bool
        True if ``mesh_ctx`` should hold this case's comparison mesh.
    """
    if reports is None or not reports.enabled or not reports.visuals:
        return False
    allow = reports.visual_case_ids
    if allow is None:
        return True
    return case_id in allow


def _normalize_metrics_config(metrics: list[str] | list[dict]) -> list[tuple[str, dict]]:
    """
    Normalize the ``metrics`` config section to ``(name, kwargs)`` pairs.

    Parameters
    ----------
    metrics : list
        Strings or dicts with a ``"name"`` key.

    Returns
    -------
    list of tuple
        ``(metric_name, kwargs_dict)`` for each entry.

    Raises
    ------
    ValueError
        If an entry is not a string or a dict with ``name``.
    """
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
    """
    Return ``surface`` or ``volume`` for this model, using registry default if unset.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration (optional ``inference_domain`` override).

    Returns
    -------
    str
        Either ``"surface"`` or ``"volume"``.
    """
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
    """
    Write ``inference_<model>_<case>.vtp`` or ``.vtu`` when requested.

    Predictions are written under VTK names from ``output_config``; ground truth
    is not required for this file.

    Parameters
    ----------
    run_config : RunConfig
        Must have ``save_inference_mesh`` True to write.
    model_config : ModelConfig
        Model name and domain.
    output_config : OutputConfig
        Mesh field name maps for surface or volume.
    wrapper : object
        Loaded model wrapper (``output_location`` selects point vs cell data).
    case : object
        Case with ``mesh_path`` and ``inference_domain``.
    case_id : str
        Case identifier for the filename.
    predictions : dict
        Decoded prediction arrays by canonical key.
    output_dir : str
        Benchmark output directory.
    dataset_name : str
        Name used in log messages.
    """
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
    """
    Invoke a registered metric, passing extended kwargs when supported.

    Falls back to ``fn(gt, predictions, **mkwargs)`` for legacy signatures.

    Parameters
    ----------
    fn : callable
        Registered metric function.
    gt : dict
        Ground-truth fields.
    predictions : dict
        Model predictions.
    case : object
        Canonical case object from the adapter.
    comparison_mesh : object or None
        PyVista mesh with GT and prediction arrays, if built.
    metric_dtype : str or None
        Element dtype label for mesh-based metrics.
    output : OutputConfig
        Output / field name configuration.
    mkwargs : dict
        Per-metric kwargs from config.

    Returns
    -------
    float or dict
        Scalar metric or dict of sub-keys (expanded by the engine).
    """
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
    shard: tuple[int, int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Run one model on one dataset: load cases, infer, compute metrics, aggregate means.

    Respects ``run.metrics_cache`` for per-case skips. Lazy-loads the model
    wrapper on the first cache miss.

    Parameters
    ----------
    model_config : ModelConfig
        Model name, checkpoint, and kwargs.
    dataset_config : DatasetConfig
        Adapter name, root, split, and case list.
    metric_names : list of tuple
        Normalized ``(metric_name, kwargs)`` pairs.
    device : str
        Torch device string for inference.
    output_dir : str
        Directory for artifacts and optional meshes.
    case_ids : list of str or None
        Cases to run; ``None`` uses ``adapter.list_cases``.
    output_config : OutputConfig
        VTK field name mappings.
    run_config : RunConfig
        Device, inference mesh export, metrics cache, etc.
    reports : ReportsConfig or None
        Optional mesh save and visual retention policy.
    allow_skip_mismatch : bool, optional
        If True, return a skipped result when surface/volume domains disagree.
    shard : tuple of (int, int) or None, optional
        If set, ``(rank, world_size)`` — keep only ``cases[rank::world_size]`` for distributed runs.

    Returns
    -------
    tuple of (dict, dict)
        Result dict with ``model``, ``dataset``, ``cases``, ``metrics``, ``per_case``,
        and ``mesh_ctx`` mapping case id -> comparison mesh for visuals.
    """
    adapter_class = get_adapter(dataset_config.name)
    m_dom = _effective_inference_domain(model_config)
    d_dom = adapter_class.inference_domain_from_kwargs(dataset_config.kwargs)
    if m_dom != d_dom:
        reason = (
            f"inference_domain mismatch: model expects {m_dom!r}, "
            f"dataset adapter {dataset_config.name!r} is {d_dom!r}"
        )
        if allow_skip_mismatch:
            log_dataset(
                "benchmark",
                f"SKIP {model_config.name!r} × {dataset_config.name!r}: {reason}",
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
    if shard is not None:
        rank, world_size = shard
        if world_size > 1:
            cases = cases[rank::world_size]
            log_dataset(
                dataset_config.name,
                f"Distributed shard: {len(cases)} case(s) for rank {rank}/{world_size}.",
            )
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

    cache_root = resolve_metrics_cache_root(
        enabled=run_config.metrics_cache.enabled,
        path=run_config.metrics_cache.path,
        output_dir=output_dir,
    )
    fingerprint: str | None = None
    if cache_root is not None:
        fingerprint = metrics_cache_fingerprint(
            model_name=model_config.name,
            model_checkpoint=model_config.checkpoint,
            model_stats_path=model_config.stats_path,
            model_kwargs=dict(model_config.kwargs),
            model_inference_domain=model_config.inference_domain,
            dataset_name=dataset_config.name,
            dataset_root=dataset_config.root,
            dataset_split=dataset_config.split,
            dataset_kwargs_resolved=dict(dkwargs),
            output_dict=output_config_to_fingerprint_dict(output_config),
            metric_specs=metric_names,
        )
        log_dataset(
            dataset_config.name,
            f"Metrics cache enabled under {cache_root} (fingerprint {fingerprint[:12]}…)…",
        )

    wrapper_class = get_model_wrapper(model_config.name)
    wrapper = None

    per_case = []
    all_metric_values: dict[str, list[float]] = {}
    mesh_ctx: dict[str, Any] = {}

    log_dataset(
        dataset_config.name,
        f"Loading {len(cases)} case(s) from root {dataset_config.root!r} "
        f"(model {model_config.name!r})…",
    )
    for cid in cases:
        cache_file = (
            metrics_cache_file_path(cache_root, fingerprint, cid)
            if cache_root is not None and fingerprint is not None
            else None
        )
        if cache_file is not None:
            blob = read_metrics_cache(cache_file)
            if (
                blob is not None
                and blob.get("fingerprint") == fingerprint
                and blob.get("model") == model_config.name
                and blob.get("dataset") == dataset_config.name
                and blob.get("case_id") == cid
            ):
                cached_metrics = metrics_from_cache_json(blob.get("metrics"))
                if cached_metrics is not None:
                    for mkey, val in cached_metrics.items():
                        all_metric_values.setdefault(mkey, []).append(val)
                    row_cb: dict[str, Any] = {"case_id": cid, "metrics": cached_metrics}
                    md_b = blob.get("metric_dtype")
                    if md_b:
                        row_cb["metric_dtype"] = md_b
                    cmp_b = blob.get("comparison_mesh_path")
                    if cmp_b:
                        row_cb["comparison_mesh_path"] = cmp_b
                    per_case.append(row_cb)
                    log_dataset(
                        dataset_config.name,
                        f"Metrics cache hit for case {cid!r} (skipped I/O and inference).",
                    )
                    continue

        if wrapper is None:
            wrapper = wrapper_class()
            wrapper.load(
                checkpoint_path=model_config.checkpoint,
                stats_path=model_config.stats_path,
                device=device,
                **model_config.merged_kwargs_for_load(),
            )

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
                fn = get_metric(mname, domain=m_dom)
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
                if _retain_comparison_mesh_for_visual_context(reports, cid):
                    mesh_ctx[cid] = comparison_mesh
        per_case.append(row)
        if cache_file is not None and fingerprint is not None:
            try:
                write_metrics_cache(
                    cache_file,
                    fingerprint=fingerprint,
                    model=model_config.name,
                    dataset=dataset_config.name,
                    case_id=cid,
                    case_metrics=case_metrics,
                    metric_dtype=row.get("metric_dtype"),
                    comparison_mesh_path=row.get("comparison_mesh_path"),
                )
            except OSError as ex:
                log_dataset(
                    dataset_config.name,
                    f"Could not write metrics cache for case {cid!r}: {ex}",
                )

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
    case_id_override: str | list[str] | None,
) -> list[str] | None:
    """
    Resolve which case IDs to evaluate for one benchmark invocation.

    Parameters
    ----------
    dataset_case_ids : list of str or None
        Cases from dataset config (or ``None`` for “all cases” upstream).
    case_id_override : str, list of str, or None
        Hydra ``case_id`` / CLI: one case, a list (same for each dataset in
        matrix mode), or ``None`` for ``dataset_case_ids``.

    Returns
    -------
    list of str or None
        Effective case list for the run.
    """
    if case_id_override is None:
        return dataset_case_ids
    if isinstance(case_id_override, str):
        return [case_id_override] if case_id_override else dataset_case_ids
    out = [str(x) for x in case_id_override if x is not None and str(x) != ""]
    return out if out else dataset_case_ids


def run_benchmark(
    config: Config,
    *,
    case_id: str | list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Execute the benchmark from a loaded ``Config``.

    Writes JSON/CSV/HTML under ``run.output_dir``, optional artifacts, and runs
    report plugins when configured.

    Parameters
    ----------
    config : Config
        Full evaluation configuration.
    case_id : str, list of str, or None, optional
        One case, a list (reused for every dataset in matrix mode), or ``None``
        for each dataset's ``case_ids`` (or all adapter cases).

    Returns
    -------
    list of dict
        One result dict per model×dataset pair (or single pair in ``single`` mode).
    """
    import physicsnemo.cfd.evaluation.datasets.adapters  # noqa: F401
    import physicsnemo.cfd.evaluation.inference.wrappers  # noqa: F401
    import physicsnemo.cfd.evaluation.metrics  # noqa: F401 — registers built-in metrics

    metric_specs = _normalize_metrics_config(config.metrics)
    dm = try_get_distributed_manager()
    shard = shard_tuple(dm, config.run.distributed)
    device = effective_device_str(dm, config.run.device)
    log_distributed_context(dm, shard)
    is_rank0 = dm is None or int(dm.rank) == 0
    output_dir = config.run.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if is_rank0 and config.benchmark.reproducibility.log_env:
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
            shard=shard,
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
                    shard=shard,
                )
                results.append(res)
                meshes_by_run.append(mesh_ctx)

    if dm is not None and dm.world_size > 1 and config.run.distributed:
        results, meshes_by_run = gather_merge_benchmark_outputs(dm, results, meshes_by_run)

    if is_rank0 and config.benchmark.reproducibility.save_artifacts:
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

    if is_rank0:
        write_report(results, output_dir, formats=["json", "csv", "html"])

    if is_rank0 and config.reports.enabled and config.reports.visuals:
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
    """
    Convert ``Config`` to a JSON-serializable dict for ``benchmark_artifacts.json``.

    Parameters
    ----------
    c : Config
        Active configuration.

    Returns
    -------
    dict
        Nested mapping suitable for ``json.dump``.
    """
    return {
        "run": {
            "device": c.run.device,
            "output_dir": c.run.output_dir,
            "seed": c.run.seed,
            "batch_size": c.run.batch_size,
            "save_inference_mesh": c.run.save_inference_mesh,
            "metrics_cache": {
                "enabled": c.run.metrics_cache.enabled,
                "path": c.run.metrics_cache.path,
            },
            "distributed": c.run.distributed,
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
            "visual_case_ids": c.reports.visual_case_ids,
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
