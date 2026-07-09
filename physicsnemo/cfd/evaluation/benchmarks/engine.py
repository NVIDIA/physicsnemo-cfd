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
replace mesh or visualization workflows. The cache fingerprint includes
``run.seed`` (influences subsampling / ``randperm`` RNG in model wrappers).

When ``save_inference_mesh`` is enabled, per-case ``inference_<model>_<case>.vt[p|u]`` meshes are
written for every scored case unless ``reports.visual_case_ids`` is set — in which case only those
ids get a mesh (all other cases are still scored for metrics). This keeps large validation runs from
dumping one VTP per case when only a couple are needed for inspection / report visuals.

When ``save_inference_mesh`` is enabled but exporting ``inference_<model>_<case>.vt[p|u]`` fails,
the full traceback is logged and persisted for audit: ``per_case[]`` keys and ``benchmark_artifacts.json``
(``inference_mesh_write_failures``, ``comparison_mesh_build_failures``, ``comparison_mesh_save_failures``)
when reproducibility artifacts are saved.

Multi-GPU: launch with ``torchrun`` (or any launcher that sets ``WORLD_SIZE`` /
``LOCAL_RANK``) so ``physicsnemo.distributed.DistributedManager`` initializes.
With ``run.distributed`` true (default) and world size > 1, cases are strided
across ranks (``cases[rank::world_size]``), results are merged on rank 0, then
broadcast to all ranks; JSON/CSV/HTML, artifacts, and report plugins run on
rank 0 only. Inference uses ``str(dm.device)`` per rank when ``DistributedManager`` is active.
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import torch


class BenchmarkPolicyError(RuntimeError):
    """Raised when :func:`run_benchmark` policy flags reject the result (e.g. all runs skipped)."""


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
from physicsnemo.cfd.evaluation.datasets.gt_alignment import (
    resolve_dataset_kwargs_for_model,
)
from physicsnemo.cfd.evaluation.assets import resolve_model_assets
from physicsnemo.cfd.evaluation.common.inference_seed import seed_inference_rng
from physicsnemo.cfd.evaluation.common.natural_sort import natural_sorted
from physicsnemo.cfd.evaluation.datasets.progress import log_dataset
from physicsnemo.cfd.evaluation.datasets.schema import (
    FieldDistribution,
    distribution_mean,
    normalize_inference_domain_str,
)
from physicsnemo.cfd.evaluation.models import get_model_wrapper
from physicsnemo.cfd.evaluation.models.model_registry import (
    get_inference_domain_for_model,
)
from physicsnemo.cfd.evaluation.metrics import get_metric
from physicsnemo.cfd.evaluation.metrics.mesh_bridge import build_comparison_mesh
from physicsnemo.cfd.postprocessing_tools.metric_registry import (
    is_reducer_metric,
    is_sample_metric,
)
from physicsnemo.cfd.evaluation.benchmarks.uq_inference import (
    compute_sparsification_payload,
    finalize_reducer_metrics,
    finalize_sample_metrics,
    is_uq_partial_key,
    make_reducer_partial_key,
    make_sample_partial_key,
    run_sampling_inference,
    strip_reducer_partials,
)

# Recoverable VTK / NumPy / mesh_bridge failures. Avoid bare ``except Exception`` —
# unexpected subclasses propagate so regressions are not mistaken for metric NaNs.
_MESH_IO_BRIDGE_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    RuntimeError,
    MemoryError,
)


def _pyvista_metric_recovery_types() -> tuple[type[BaseException], ...]:
    """Subclasses raised by PyVista during metric mesh operations ( VTK / IO )."""
    try:
        import pyvista as pv  # noqa: PLC0415
    except ImportError:
        return ()
    names = (
        "AmbiguousDataError",
        "InvalidMeshError",
        "MissingDataError",
        "VTKExecutionError",
        "VTKVersionError",
        "PointSetCellOperationError",
        "PyVistaAttributeError",
        "PyVistaPipelineError",
        "NotAllTrianglesError",
    )
    tt: list[type[BaseException]] = []
    for name in names:
        obj = getattr(pv, name, None)
        if isinstance(obj, type) and issubclass(obj, BaseException):
            tt.append(obj)
    return tuple(tt)


# Only while running the callable returned by ``get_metric`` — **not** lookup failures:
# ``KeyError`` from :func:`~physicsnemo.cfd.evaluation.metrics.get_metric` propagates so
# unregistered names / domain mismatches fail loudly.
_METRIC_COMPUTE_RECOVERABLE: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    IndexError,
    ArithmeticError,
    OSError,
    MemoryError,
) + _pyvista_metric_recovery_types()


def _audit_traceback_entries(
    results: list[dict[str, Any]],
    per_case_tb_key: str,
) -> list[dict[str, Any]]:
    """Collect non-empty traceback strings on ``per_case`` rows for ``benchmark_artifacts.json``."""
    out: list[dict[str, Any]] = []
    for r in results:
        if r.get("skipped"):
            continue
        for row in r.get("per_case") or []:
            tb = row.get(per_case_tb_key)
            if isinstance(tb, str) and tb.strip():
                out.append(
                    {
                        "model": r["model"],
                        "dataset": r["dataset"],
                        "case_id": row["case_id"],
                        "traceback": tb,
                    }
                )
    return out


def _retain_comparison_mesh_for_visual_context(
    reports: ReportsConfig | None, case_id: str
) -> bool:
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


def _save_inference_mesh_for_case(
    reports: ReportsConfig | None, case_id: str
) -> bool:
    """Whether this case should write an ``inference_<model>_<case>`` mesh.

    Saving every case's mesh is wasteful for large validation sets. When
    ``reports.visual_case_ids`` is set, restrict inference-mesh writes to exactly those cases
    (the ones you inspect / that the report visuals use); all other cases are scored for
    metrics only. When it is ``None`` there is no restriction (write every case, back-compat).
    """
    if reports is None or reports.visual_case_ids is None:
        return True
    return case_id in reports.visual_case_ids


def _normalize_metrics_config(
    metrics: list[str | dict[str, Any]],
) -> list[tuple[str, dict]]:
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
    """Resolve ``surface``/``volume`` for adapters, metrics, and cache (Hydra + wrappers)."""
    kw = model_config.merged_kwargs_for_load()
    dom_kw = kw.get("inference_domain")
    if dom_kw is not None:
        return dom_kw

    cls = get_model_wrapper(model_config.name)
    hinted = cls.inference_domain_from_kwargs(dict(kw))
    if hinted is not None:
        return normalize_inference_domain_str(
            hinted if isinstance(hinted, str) else str(hinted),
            parameter=f"{cls.__name__}.inference_domain_from_kwargs()",
        )

    return get_inference_domain_for_model(model_config.name)


def _save_inference_mesh_if_requested(
    *,
    run_config: RunConfig,
    model_config: ModelConfig,
    output_config: OutputConfig,
    reports: ReportsConfig | None,
    wrapper: Any,
    case: Any,
    case_id: str,
    predictions: dict[str, Any],
    output_dir: str,
    dataset_name: str,
) -> str | None:
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

    Returns
    -------
    str or None
        ``None`` if writing was skipped or succeeded. On failure, a full traceback string
        for logging and ``benchmark_artifacts.json`` / per-case results.
    """
    if not run_config.save_inference_mesh:
        return None
    if not _save_inference_mesh_for_case(reports, case_id):
        return None
    import pyvista as pv

    m_dom = case.inference_domain
    out_path = (
        Path(output_dir)
        / f"inference_{model_config.name}_{case_id}{'.vtp' if m_dom == 'surface' else '.vtu'}"
    )
    log_dataset(
        dataset_name,
        f"Writing inference mesh (predictions only) to {out_path}…",
    )
    try:
        ref_geo = getattr(case, "reference_geometry", None)
        if m_dom == "surface":
            if ref_geo is not None:
                mesh = ref_geo
                if not isinstance(mesh, pv.PolyData):
                    mesh = mesh.extract_surface()
            else:
                mesh = pv.read(case.mesh_path)
                if not isinstance(mesh, pv.PolyData):
                    mesh = mesh.extract_surface()
            names = output_config.mesh_field_names
        else:
            if ref_geo is not None:
                mesh = ref_geo
                if hasattr(mesh, "cast_to_unstructured_grid"):
                    mesh = mesh.cast_to_unstructured_grid()
            else:
                mesh = pv.read(case.mesh_path)
                if hasattr(mesh, "cast_to_unstructured_grid"):
                    mesh = mesh.cast_to_unstructured_grid()
            names = output_config.volume_mesh_field_names

        if m_dom == "surface":
            std_names = output_config.std_mesh_field_names
            epi_names = output_config.epistemic_std_mesh_field_names
        else:
            std_names = output_config.std_volume_mesh_field_names
            epi_names = output_config.epistemic_std_volume_mesh_field_names

        data_target = (
            mesh.cell_data if wrapper.output_location == "cell" else mesh.point_data
        )
        for canonical_key, mesh_name in names.items():
            if canonical_key not in predictions:
                continue
            value = predictions[canonical_key]
            data_target[mesh_name] = distribution_mean(value)
            # Attach uncertainty companions so exported meshes carry UQ for ParaView / visuals.
            if isinstance(value, FieldDistribution):
                if value.std is not None:
                    data_target[std_names.get(canonical_key) or f"{mesh_name}Std"] = (
                        value.std
                    )
                if value.epistemic_std is not None:
                    data_target[
                        epi_names.get(canonical_key) or f"{mesh_name}EpistemicStd"
                    ] = value.epistemic_std
        mesh.save(str(out_path))
        log_dataset(dataset_name, f"Wrote inference mesh: {out_path}")
    except _MESH_IO_BRIDGE_ERRORS:
        tb = traceback.format_exc()
        log_dataset(
            dataset_name,
            f"Could not write inference mesh to {out_path}:\n{tb}",
        )
        return tb
    return None


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


def _call_reducer_partial(
    metric: Any,
    gt: dict,
    predictions: dict,
    *,
    case: Any,
    comparison_mesh: Any,
    metric_dtype: str | None,
    output: OutputConfig,
    mkwargs: dict[str, Any],
) -> dict[str, float]:
    """Invoke a reducer metric's ``partial`` with extended kwargs, falling back to the basics.

    Returns the per-case extensive sufficient statistics (additive across cases).
    """
    extended = dict(mkwargs)
    extended.update(
        case=case,
        comparison_mesh=comparison_mesh,
        metric_dtype=metric_dtype,
        output=output,
    )
    try:
        return metric.partial(gt, predictions, **extended)
    except TypeError:
        return metric.partial(gt, predictions, **mkwargs)


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
    case_cache: dict[tuple, Any] | None = None,
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
        Adapter name, root, and case list.
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
    case_cache : dict, optional
        Shared mutable mapping keyed by ``(dataset_name, dataset_root, resolved_dkwargs, case_id)``.
        When provided (matrix mode), ``adapter.load_case(cid)`` is called once per unique key
        across all models in the matrix; subsequent invocations reuse the cached
        :class:`~physicsnemo.cfd.evaluation.datasets.schema.CanonicalCase` (and its
        ``reference_geometry``) instead of re-reading the VTU/VTP from disk.

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
    # Stable key fragment shared by all cases of this (model, dataset) — appended with cid
    # below to look up / populate the matrix-level case cache.
    case_cache_dataset_key: tuple[Any, ...] = (
        dataset_config.name,
        dataset_config.root,
        json.dumps(dkwargs, sort_keys=True, default=str),
    )
    log_dataset(
        dataset_config.name,
        f"Listing cases under root {dataset_config.root!r}…",
    )
    cases = case_ids if case_ids is not None else adapter.list_cases()
    cases = natural_sorted(cases)
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

    wrapper_class = get_model_wrapper(model_config.name)
    resolved_ck, resolved_st, asset_identity, asset_load_kw = resolve_model_assets(
        model_config, wrapper_class
    )
    fp_ck = "" if asset_identity else resolved_ck
    fp_st = "" if asset_identity else resolved_st

    cache_root = resolve_metrics_cache_root(
        enabled=run_config.metrics_cache.enabled,
        path=run_config.metrics_cache.path,
        output_dir=output_dir,
    )
    fingerprint: str | None = None
    if cache_root is not None:
        fingerprint = metrics_cache_fingerprint(
            model_name=model_config.name,
            model_checkpoint=fp_ck,
            model_stats_path=fp_st,
            model_kwargs=dict(model_config.kwargs),
            model_inference_domain=m_dom,
            model_asset_identity=asset_identity,
            dataset_name=dataset_config.name,
            dataset_root=dataset_config.root,
            dataset_kwargs_resolved=dict(dkwargs),
            output_dict=output_config_to_fingerprint_dict(output_config),
            metric_specs=metric_names,
            run_seed=run_config.seed,
            run_uq={
                "enabled": run_config.uq.enabled,
                "num_samples": run_config.uq.num_samples,
                "retain_samples": run_config.uq.retain_samples,
            },
        )
        log_dataset(
            dataset_config.name,
            f"Metrics cache enabled under {cache_root} (fingerprint {fingerprint[:12]}…)…",
        )

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
            load_kw = {**asset_load_kw, **model_config.merged_kwargs_for_load()}
            wrapper.load(
                checkpoint_path=resolved_ck,
                stats_path=resolved_st,
                device=device,
                **load_kw,
            )

        case_key = (*case_cache_dataset_key, cid)
        if case_cache is not None and case_key in case_cache:
            case = case_cache[case_key]
            log_dataset(
                dataset_config.name,
                f"Reusing cached case {cid!r} (skipping VTU/VTP read).",
            )
        else:
            log_dataset(
                dataset_config.name,
                f"Reading case {cid!r}…",
            )
            case = adapter.load_case(cid)
            if case_cache is not None:
                case_cache[case_key] = case
        seed_inference_rng(run_config.seed, cid)
        model_input = wrapper.prepare_inputs(case)
        supports_uq = bool(getattr(wrapper, "SUPPORTS_UQ", False))
        uq_method = getattr(wrapper, "UQ_METHOD", "none")
        if supports_uq and uq_method == "sampling" and run_config.uq.enabled:
            # N stochastic passes; prepare_inputs already ran once (see UQ design doc §6.2).
            predictions = run_sampling_inference(
                wrapper,
                case,
                model_input,
                n=run_config.uq.num_samples,
                run_seed=run_config.seed,
                case_id=cid,
                retain_samples=run_config.uq.retain_samples,
            )
        elif supports_uq and uq_method == "analytic":
            raw = wrapper.predict(model_input)
            predictions = wrapper.decode_distribution(raw, case, model_input)
        else:
            raw = wrapper.predict(model_input)
            predictions = wrapper.decode_outputs(raw, case, model_input)
        gt = case.ground_truth or {}

        inference_mesh_err = _save_inference_mesh_if_requested(
            run_config=run_config,
            model_config=model_config,
            output_config=output_config,
            reports=reports,
            wrapper=wrapper,
            case=case,
            case_id=cid,
            predictions=predictions,
            output_dir=output_dir,
            dataset_name=dataset_config.name,
        )

        comparison_mesh = None
        metric_dtype: str | None = None
        comparison_mesh_build_err: str | None = None
        try:
            comparison_mesh, metric_dtype = build_comparison_mesh(
                case, predictions, output_config, mesh_override=case.reference_geometry
            )
        except _MESH_IO_BRIDGE_ERRORS:
            comparison_mesh_build_err = traceback.format_exc()
            log_dataset(
                dataset_config.name,
                f"Warning: comparison mesh not built for case {cid!r}:\n{comparison_mesh_build_err}",
            )

        case_metrics: dict[str, float] = {}
        for mname, mkwargs in metric_names:
            metric_fn = get_metric(mname, domain=m_dom)
            try:
                if is_sample_metric(metric_fn):
                    # Sample metric: store per-geometry scalars under a reserved key. Collected
                    # (not summed) across cases + ranks and finalized after the loop / merge.
                    partial_stats = _call_reducer_partial(
                        metric_fn,
                        gt,
                        predictions,
                        case=case,
                        comparison_mesh=comparison_mesh,
                        metric_dtype=metric_dtype,
                        output=output_config,
                        mkwargs=mkwargs,
                    )
                    for pkey, pval in partial_stats.items():
                        rk = make_sample_partial_key(mname, pkey)
                        case_metrics[rk] = float(pval)
                        all_metric_values.setdefault(rk, []).append(float(pval))
                    continue
                if is_reducer_metric(metric_fn):
                    # Pooled metric: store per-case additive sufficient statistics under a
                    # reserved key so they cache + merge like scalars; finalized after the loop.
                    partial_stats = _call_reducer_partial(
                        metric_fn,
                        gt,
                        predictions,
                        case=case,
                        comparison_mesh=comparison_mesh,
                        metric_dtype=metric_dtype,
                        output=output_config,
                        mkwargs=mkwargs,
                    )
                    for pkey, pval in partial_stats.items():
                        rk = make_reducer_partial_key(mname, pkey)
                        case_metrics[rk] = float(pval)
                        all_metric_values.setdefault(rk, []).append(float(pval))
                    continue
                out = _call_metric(
                    metric_fn,
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
            except _METRIC_COMPUTE_RECOVERABLE:
                mtb = traceback.format_exc()
                log_dataset(
                    dataset_config.name,
                    f"Metric {mname!r} recoverable failure for {cid!r} (NaN recorded):\n{mtb}",
                )
                case_metrics[mname] = float("nan")
                all_metric_values.setdefault(mname, []).append(float("nan"))
            except Exception:
                mtb = traceback.format_exc()
                log_dataset(
                    dataset_config.name,
                    f"Metric {mname!r} failed for {cid!r} (non-recoverable; re-raising):\n{mtb}",
                )
                raise
        row: dict[str, Any] = {"case_id": cid, "metrics": case_metrics}
        if inference_mesh_err:
            row["inference_mesh_write_error"] = inference_mesh_err
        if comparison_mesh_build_err:
            row["comparison_mesh_build_error"] = comparison_mesh_build_err
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
                    except _MESH_IO_BRIDGE_ERRORS:
                        save_tb = traceback.format_exc()
                        log_dataset(
                            dataset_config.name,
                            f"Could not save comparison mesh for {cid!r}:\n{save_tb}",
                        )
                        row["comparison_mesh_save_error"] = save_tb
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

    # Aggregate (mean over cases) for pointwise metrics; reducer / sample partials (reserved
    # keys) are finalized separately below (pooled over points, resp. collected over geometries).
    metrics_summary = {}
    for mname, values in all_metric_values.items():
        if is_uq_partial_key(mname):
            continue
        valid = [v for v in values if v == v]  # filter nan
        metrics_summary[mname] = sum(valid) / len(valid) if valid else float("nan")

    # Pooled reducer + sample-wise metrics. In distributed runs these are recomputed after the
    # merge on rank 0 (merge_benchmark_result_shards) from the merged per-case partials; this
    # local pass keeps single-process runs correct.
    metrics_summary.update(finalize_reducer_metrics(per_case, m_dom))
    metrics_summary.update(finalize_sample_metrics(per_case, m_dom))

    return (
        {
            "model": model_config.name,
            "dataset": dataset_config.name,
            "cases": cases,
            "metrics": metrics_summary,
            "per_case": per_case,
            "inference_domain": m_dom,
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

    Raises
    ------
    BenchmarkPolicyError
        If ``run.fail_on_all_skipped`` or ``run.fail_on_any_metric_nan`` rejects the outcome.
    """
    import physicsnemo.cfd.evaluation.datasets.adapters  # noqa: F401
    import physicsnemo.cfd.evaluation.models.wrappers  # noqa: F401
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
        # Single read per (dataset, dkwargs, case_id) across all models. Volume VTUs are
        # tens of GiB; without this, each model's adapter re-reads the same file and the
        # in-flight read coexists in RAM with the previous model's retained ``mesh_ctx``.
        matrix_case_cache: dict[tuple, Any] = {}
        for m_cfg in models:
            for d_cfg in datasets:
                # Free residual wrapper / dataset state from the previous matrix iteration so the
                # next model starts clean (Python refs in mesh_ctx etc. can otherwise hold model
                # weights and per-case tensors alive on host RAM and the device).
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
                    case_cache=matrix_case_cache,
                )
                results.append(res)
                meshes_by_run.append(mesh_ctx)
        matrix_case_cache.clear()

    if dm is not None and dm.world_size > 1 and config.run.distributed:
        results, meshes_by_run = gather_merge_benchmark_outputs(
            dm, results, meshes_by_run
        )

    # Reducer sufficient statistics (reserved ``_uq::`` keys) have been folded into each run's
    # ``metrics`` summary; drop them from the reported per-case rows so outputs show real values.
    # Before stripping, harvest the sample-metric per-geometry curves for the sparsification visual.
    # These are kept in a side list aligned with ``results`` (not on the result dicts) so the numpy
    # curve arrays never reach the JSON/CSV/HTML report serializers.
    sparsification_by_run: list[dict[str, Any]] = []
    for r in results:
        if r.get("skipped"):
            sparsification_by_run.append({})
            continue
        sparsification_by_run.append(
            compute_sparsification_payload(
                r.get("per_case") or [], r.get("inference_domain")
            )
        )
    for r in results:
        strip_reducer_partials(r.get("per_case") or [])

    if is_rank0 and config.benchmark.reproducibility.save_artifacts:
        artifacts = Path(output_dir) / "benchmark_artifacts.json"
        skipped = [r for r in results if r.get("skipped")]
        log_dataset("benchmark", f"Writing artifacts to {artifacts}…")
        with open(artifacts, "w") as f:
            inference_failures = _audit_traceback_entries(
                results, "inference_mesh_write_error"
            )
            cmp_build_failures = _audit_traceback_entries(
                results, "comparison_mesh_build_error"
            )
            cmp_save_failures = _audit_traceback_entries(
                results, "comparison_mesh_save_error"
            )
            payload: dict[str, Any] = {
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
            }
            if inference_failures:
                payload["inference_mesh_write_failures"] = inference_failures
            if cmp_build_failures:
                payload["comparison_mesh_build_failures"] = cmp_build_failures
            if cmp_save_failures:
                payload["comparison_mesh_save_failures"] = cmp_save_failures
            json.dump(payload, f, indent=2)

    if is_rank0:
        write_report(results, output_dir, formats=["json", "csv", "html"])

    if is_rank0 and config.reports.enabled and config.reports.visuals:
        import physicsnemo.cfd.evaluation.reports  # noqa: F401 — register built-in visuals

        from physicsnemo.cfd.evaluation.benchmarks.report_plugins import (
            run_optional_report_plugins,
        )

        log_dataset("benchmark", "Running reports.visuals from benchmark results…")
        run_optional_report_plugins(
            config,
            results,
            output_dir,
            context={
                "comparison_meshes_by_run": meshes_by_run,
                "uq_sparsification_by_run": sparsification_by_run,
            },
        )

    _enforce_benchmark_policy(config, results)

    return results


def run_benchmark_cli(
    config: Config,
    *,
    case_id: str | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run benchmarks for interactive / CLI callers.

    Same as :func:`run_benchmark`, but catches :class:`BenchmarkPolicyError`,
    prints the message to stderr, and terminates the process with exit code ``1``.
    Libraries and tests should call :func:`run_benchmark` directly to handle or propagate
    the exception.
    """
    try:
        return run_benchmark(config, case_id=case_id)
    except BenchmarkPolicyError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None


def _enforce_benchmark_policy(config: Config, results: list[dict[str, Any]]) -> None:
    """Raise :class:`BenchmarkPolicyError` when ``run.fail_on_*`` flags apply."""
    if not results:
        return
    run = config.run
    if run.fail_on_all_skipped and all(r.get("skipped") for r in results):
        raise BenchmarkPolicyError(
            "All benchmark runs were skipped; set run.fail_on_all_skipped=false to allow exit 0, "
            "or fix model/dataset domain and paths."
        )
    if run.fail_on_any_metric_nan:
        for r in results:
            if r.get("skipped"):
                continue
            metrics = r.get("metrics") or {}
            for _k, v in metrics.items():
                if isinstance(v, float) and math.isnan(v):
                    raise BenchmarkPolicyError(
                        "Aggregate metric NaN encountered; set run.fail_on_any_metric_nan=false to allow exit 0, "
                        "or fix failing metrics."
                    )


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
            "uq": {
                "enabled": c.run.uq.enabled,
                "num_samples": c.run.uq.num_samples,
                "retain_samples": c.run.uq.retain_samples,
                "device_metrics": c.run.uq.device_metrics,
            },
            "distributed": c.run.distributed,
            "fail_on_all_skipped": c.run.fail_on_all_skipped,
            "fail_on_any_metric_nan": c.run.fail_on_any_metric_nan,
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
