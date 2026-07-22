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

"""Helpers for multi-GPU benchmark runs using PhysicsNeMo ``DistributedManager``."""

from __future__ import annotations

import os
from typing import Any

import torch.distributed as dist

from physicsnemo.cfd.evaluation.common.natural_sort import natural_sort_key
from physicsnemo.cfd.evaluation.datasets.progress import log_dataset


def try_get_distributed_manager() -> Any | None:
    """
    Return an initialized ``DistributedManager`` instance, or ``None`` for single-process runs.

    Raises
    ------
    RuntimeError
        If ``WORLD_SIZE`` > 1 but ``physicsnemo.distributed`` is unavailable.
    """
    try:
        from physicsnemo.distributed import DistributedManager
    except ImportError as e:
        if int(os.environ.get("WORLD_SIZE", "1") or "1") > 1:
            raise RuntimeError(
                "Multi-process launch detected (WORLD_SIZE>1) but physicsnemo.distributed is not installed."
            ) from e
        return None
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    return DistributedManager()


def shard_tuple(dm: Any | None, distributed_enabled: bool) -> tuple[int, int] | None:
    """Return ``(rank, world_size)`` for case sharding, or ``None`` when sharding is off."""
    if not distributed_enabled or dm is None or dm.world_size <= 1:
        return None
    return (int(dm.rank), int(dm.world_size))


def effective_device_str(dm: Any | None, run_device: str) -> str:
    """Use ``DistributedManager`` device when available; otherwise ``run.device``."""
    if dm is None:
        return run_device
    return str(dm.device)


def merge_benchmark_result_shards(shards: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Combine per-rank benchmark ``dict``s for the same model×dataset into one result.

    Recomputes aggregate ``metrics`` from the union of ``per_case`` rows (mean over cases,
    ignoring NaNs per key). Sorts ``per_case`` by ``case_id`` (natural / alphanumeric).
    """
    if not shards:
        return {}
    if all(s.get("skipped") for s in shards):
        return dict(shards[0])
    active = [s for s in shards if not s.get("skipped")]
    if not active:
        return dict(shards[0])
    model = active[0]["model"]
    dataset = active[0]["dataset"]
    for s in active[1:]:
        if s["model"] != model or s["dataset"] != dataset:
            raise RuntimeError(
                f"Distributed merge mismatch: expected {model!r}×{dataset!r}, "
                f"got {s['model']!r}×{s['dataset']!r}"
            )
    from physicsnemo.cfd.evaluation.benchmarks.uq_inference import (
        finalize_reducer_metrics,
        finalize_sample_metrics,
        is_uq_partial_key,
    )

    per_case: list[dict[str, Any]] = []
    for s in active:
        per_case.extend(s.get("per_case") or [])
    per_case.sort(key=lambda r: natural_sort_key(r.get("case_id")))
    all_metric_values: dict[str, list[float]] = {}
    for row in per_case:
        for k, v in (row.get("metrics") or {}).items():
            all_metric_values.setdefault(k, []).append(float(v))
    metrics_summary: dict[str, float] = {}
    for mname, values in all_metric_values.items():
        # Reducer / sample statistics are finalized (pooled / collected), not mean-over-cases.
        if is_uq_partial_key(mname):
            continue
        valid = [v for v in values if v == v]
        metrics_summary[mname] = sum(valid) / len(valid) if valid else float("nan")
    inference_domain = active[0].get("inference_domain")
    metrics_summary.update(finalize_reducer_metrics(per_case, inference_domain))
    metrics_summary.update(finalize_sample_metrics(per_case, inference_domain))
    # Preserve NaN placeholders for configured-but-unavailable UQ metrics: each shard already
    # finalized them (via the configured metric names in _run_single_model_dataset), so carry any
    # summary key missing from the merged set forward as-is. setdefault never clobbers a value
    # recomputed above from real partials.
    for s in active:
        for k, v in (s.get("metrics") or {}).items():
            if isinstance(v, (int, float)):
                metrics_summary.setdefault(k, float(v))
    case_ids_raw = [str(r["case_id"]) for r in per_case if r.get("case_id") is not None]
    case_ids_no_dup = sorted(set(case_ids_raw), key=natural_sort_key)
    return {
        "model": model,
        "dataset": dataset,
        "cases": case_ids_no_dup,
        "metrics": metrics_summary,
        "per_case": per_case,
        "inference_domain": inference_domain,
    }


def merge_mesh_context_shards(shards: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge rank-local comparison mesh dicts."""
    out: dict[str, Any] = {}
    for s in shards:
        out.update(s)
    return out


def gather_merge_benchmark_outputs(
    dm: Any,
    results: list[dict[str, Any]],
    meshes_by_run: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Collect per-rank ``results`` / ``meshes_by_run`` on rank 0, merge, and broadcast to all ranks.
    """
    world = int(dm.world_size)
    payload = (results, meshes_by_run)
    if world <= 1:
        return results, meshes_by_run

    if int(dm.rank) == 0:
        gather_list: list[Any | None] = [None] * world
        dist.gather_object(payload, gather_list, dst=0)
        merged_results: list[dict[str, Any]] = []
        merged_meshes: list[dict[str, Any]] = []
        first = gather_list[0]
        assert first is not None
        n_runs = len(first[0])
        for j in range(n_runs):
            res_shards = [gather_list[i][0][j] for i in range(world)]  # type: ignore[index]
            merged_results.append(merge_benchmark_result_shards(res_shards))
            mesh_shards = [gather_list[i][1][j] for i in range(world)]  # type: ignore[index]
            merged_meshes.append(merge_mesh_context_shards(mesh_shards))
        out_payload: tuple[list[dict[str, Any]], list[dict[str, Any]]] = (
            merged_results,
            merged_meshes,
        )
    else:
        dist.gather_object(payload, None, dst=0)
        out_payload = None

    dist.barrier()
    if int(dm.rank) == 0:
        obj_list: list[Any | None] = [out_payload]
    else:
        obj_list = [None]
    dist.broadcast_object_list(obj_list, src=0)
    merged_pair = obj_list[0]
    assert merged_pair is not None
    return merged_pair[0], merged_pair[1]


def log_distributed_context(dm: Any | None, shard: tuple[int, int] | None) -> None:
    """Log the current distributed rank/world-size and whether case sharding is active."""
    if dm is None:
        return
    if shard is None:
        log_dataset(
            "benchmark",
            f"Distributed: rank {dm.rank}/{dm.world_size} (no case sharding).",
        )
    else:
        r, w = shard
        log_dataset("benchmark", f"Distributed: rank {r}/{w} (case sharding enabled).")
