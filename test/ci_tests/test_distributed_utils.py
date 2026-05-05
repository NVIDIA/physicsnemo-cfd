# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for benchmark distributed helpers and ``_case_ids_for_run`` (no GPU / torchrun)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from physicsnemo.cfd.evaluation.benchmarks.distributed_utils import (
    effective_device_str,
    merge_benchmark_result_shards,
    merge_mesh_context_shards,
    shard_tuple,
)
from physicsnemo.cfd.evaluation.benchmarks.engine import _case_ids_for_run


def test_merge_benchmark_result_shards_empty() -> None:
    """Merging zero shards yields an empty dict."""
    assert merge_benchmark_result_shards([]) == {}


def test_merge_benchmark_result_shards_all_skipped() -> None:
    """If every shard is skipped the merged result keeps the skip flag and reason."""
    shards = [
        {
            "model": "m",
            "dataset": "d",
            "skipped": True,
            "skip_reason": "x",
            "cases": [],
            "metrics": {},
            "per_case": [],
        },
        {
            "model": "m",
            "dataset": "d",
            "skipped": True,
            "skip_reason": "x",
            "cases": [],
            "metrics": {},
            "per_case": [],
        },
    ]
    out = merge_benchmark_result_shards(shards)
    assert out["skipped"] is True
    assert out["skip_reason"] == "x"


def test_merge_benchmark_result_shards_two_ranks_sorts_and_means() -> None:
    """Two-shard merge naturally sorts cases and averages numeric metrics."""
    a = {
        "model": "m",
        "dataset": "d",
        "cases": ["c2"],
        "metrics": {"x": 0.0},
        "per_case": [{"case_id": "c2", "metrics": {"x": 2.0}}],
    }
    b = {
        "model": "m",
        "dataset": "d",
        "cases": ["c1"],
        "metrics": {"x": 0.0},
        "per_case": [{"case_id": "c1", "metrics": {"x": 4.0}}],
    }
    merged = merge_benchmark_result_shards([a, b])
    assert [r["case_id"] for r in merged["per_case"]] == ["c1", "c2"]
    assert merged["metrics"]["x"] == pytest.approx(3.0)
    assert merged["cases"] == ["c1", "c2"]


def test_merge_benchmark_result_shards_ignores_nan_in_mean() -> None:
    """NaN per-case metrics are excluded from the aggregated mean."""
    shards = [
        {
            "model": "m",
            "dataset": "d",
            "cases": ["a"],
            "metrics": {},
            "per_case": [{"case_id": "a", "metrics": {"x": 1.0}}],
        },
        {
            "model": "m",
            "dataset": "d",
            "cases": ["b"],
            "metrics": {},
            "per_case": [{"case_id": "b", "metrics": {"x": float("nan")}}],
        },
    ]
    merged = merge_benchmark_result_shards(shards)
    assert merged["metrics"]["x"] == pytest.approx(1.0)


def test_merge_benchmark_result_shards_model_mismatch() -> None:
    """Merging shards with mismatched ``model`` ids raises ``RuntimeError``."""
    with pytest.raises(RuntimeError, match="Distributed merge mismatch"):
        merge_benchmark_result_shards(
            [
                {"model": "m1", "dataset": "d", "per_case": []},
                {"model": "m2", "dataset": "d", "per_case": []},
            ]
        )


def test_merge_mesh_context_shards() -> None:
    """Mesh context shards merge by union with later entries winning on key collision."""
    a = {"run_1": "mesh_a"}
    b = {"run_2": "mesh_b", "run_1": "mesh_b_wins"}
    assert merge_mesh_context_shards([a, b]) == {
        "run_1": "mesh_b_wins",
        "run_2": "mesh_b",
    }


def test_effective_device_str_no_dm() -> None:
    """Without a DistributedManager the configured device is returned verbatim."""
    assert effective_device_str(None, "cuda:0") == "cuda:0"


def test_effective_device_str_uses_dm() -> None:
    """A DistributedManager-supplied device takes precedence over the configured one."""
    dm = SimpleNamespace(device="cuda:3")
    assert effective_device_str(dm, "cuda:0") == "cuda:3"


def test_shard_tuple_disabled_or_single() -> None:
    """Sharding is a no-op when disabled, when world size is 1, or when ``dm`` is missing."""
    dm = SimpleNamespace(world_size=4, rank=1)
    assert shard_tuple(dm, distributed_enabled=False) is None
    dm1 = SimpleNamespace(world_size=1, rank=0)
    assert shard_tuple(dm1, distributed_enabled=True) is None
    assert shard_tuple(None, distributed_enabled=True) is None


def test_shard_tuple_multi() -> None:
    """Multi-rank sharding returns ``(rank, world_size)``."""
    dm = SimpleNamespace(world_size=4, rank=1)
    assert shard_tuple(dm, distributed_enabled=True) == (1, 4)


def test_case_ids_for_run_none_uses_dataset() -> None:
    """``None`` override falls back to the dataset's case list (or ``None`` when empty)."""
    assert _case_ids_for_run(["a", "b"], None) == ["a", "b"]
    assert _case_ids_for_run(None, None) is None


def test_case_ids_for_run_string() -> None:
    """A string override is wrapped into a single-element list."""
    assert _case_ids_for_run(["a", "b"], "z") == ["z"]


def test_case_ids_for_run_empty_string_fallback() -> None:
    """An empty-string override falls back to the dataset's case list."""
    assert _case_ids_for_run(["a", "b"], "") == ["a", "b"]


def test_case_ids_for_run_list() -> None:
    """A list override replaces the dataset case list verbatim."""
    assert _case_ids_for_run(["a", "b"], ["x", "y"]) == ["x", "y"]


def test_case_ids_for_run_empty_list_fallback() -> None:
    """An empty-list override falls back to the dataset's case list."""
    assert _case_ids_for_run(["a", "b"], []) == ["a", "b"]
