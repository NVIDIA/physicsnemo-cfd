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
    assert merge_benchmark_result_shards([]) == {}


def test_merge_benchmark_result_shards_all_skipped() -> None:
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
    with pytest.raises(RuntimeError, match="Distributed merge mismatch"):
        merge_benchmark_result_shards(
            [
                {"model": "m1", "dataset": "d", "per_case": []},
                {"model": "m2", "dataset": "d", "per_case": []},
            ]
        )


def test_merge_mesh_context_shards() -> None:
    a = {"run_1": "mesh_a"}
    b = {"run_2": "mesh_b", "run_1": "mesh_b_wins"}
    assert merge_mesh_context_shards([a, b]) == {"run_1": "mesh_b_wins", "run_2": "mesh_b"}


def test_effective_device_str_no_dm() -> None:
    assert effective_device_str(None, "cuda:0") == "cuda:0"


def test_effective_device_str_uses_dm() -> None:
    dm = SimpleNamespace(device="cuda:3")
    assert effective_device_str(dm, "cuda:0") == "cuda:3"


def test_shard_tuple_disabled_or_single() -> None:
    dm = SimpleNamespace(world_size=4, rank=1)
    assert shard_tuple(dm, distributed_enabled=False) is None
    dm1 = SimpleNamespace(world_size=1, rank=0)
    assert shard_tuple(dm1, distributed_enabled=True) is None
    assert shard_tuple(None, distributed_enabled=True) is None


def test_shard_tuple_multi() -> None:
    dm = SimpleNamespace(world_size=4, rank=1)
    assert shard_tuple(dm, distributed_enabled=True) == (1, 4)


def test_case_ids_for_run_none_uses_dataset() -> None:
    assert _case_ids_for_run(["a", "b"], None) == ["a", "b"]
    assert _case_ids_for_run(None, None) is None


def test_case_ids_for_run_string() -> None:
    assert _case_ids_for_run(["a", "b"], "z") == ["z"]


def test_case_ids_for_run_empty_string_fallback() -> None:
    assert _case_ids_for_run(["a", "b"], "") == ["a", "b"]


def test_case_ids_for_run_list() -> None:
    assert _case_ids_for_run(["a", "b"], ["x", "y"]) == ["x", "y"]


def test_case_ids_for_run_empty_list_fallback() -> None:
    assert _case_ids_for_run(["a", "b"], []) == ["a", "b"]
