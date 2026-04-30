# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose Hydra configs for ``workflows/benchmarking_workflow`` and load :class:`Config` (no GPU / no inference)."""

from __future__ import annotations

from pathlib import Path

import pytest

from physicsnemo.cfd.evaluation.benchmarks.hydra_utils import hydra_config_to_benchmark_dict
from physicsnemo.cfd.evaluation.config import Config

pytest.importorskip("hydra")
from hydra import compose, initialize_config_dir  # noqa: E402

_BENCHMARK_CONF = (
    Path(__file__).resolve().parents[2] / "workflows" / "benchmarking_workflow" / "conf"
)


@pytest.mark.parametrize(
    "config_name",
    [
        "config_matrix_surface_hf",
        "config_matrix_volume_hf",
        "config_matrix_surface_custom",
        "config_matrix_volume_custom",
    ],
)
def test_hydra_config_loads_into_config(config_name: str) -> None:
    if not _BENCHMARK_CONF.is_dir():
        pytest.skip("benchmarking_workflow/conf not in tree")
    with initialize_config_dir(version_base="1.3", config_dir=str(_BENCHMARK_CONF)):
        cfg = compose(config_name=config_name)
    raw, case_id = hydra_config_to_benchmark_dict(cfg)
    Config.from_dict(raw)
    # case_id may be None, str, or list — normalization is covered by hydra_utils
    assert case_id is None or isinstance(case_id, (str, list))


def test_hydra_config_to_benchmark_dict_case_id_variants() -> None:
    from omegaconf import OmegaConf

    base = {"run": {"device": "cpu", "output_dir": "out"}, "metrics": ["l2_pressure"]}
    d1 = OmegaConf.create({**base, "case_id": "run_1"})
    raw, cid = hydra_config_to_benchmark_dict(d1)
    assert cid == "run_1"
    assert "case_id" not in raw


def test_benchmark_policy_fail_on_all_skipped() -> None:
    from physicsnemo.cfd.evaluation.benchmarks.engine import BenchmarkPolicyError, _enforce_benchmark_policy

    cfg = Config.from_dict(
        {
            "run": {"output_dir": "out", "fail_on_all_skipped": True},
            "metrics": ["l2_pressure"],
        }
    )
    with pytest.raises(BenchmarkPolicyError):
        _enforce_benchmark_policy(
            cfg,
            [{"skipped": True, "metrics": {}, "model": "a", "dataset": "b"}],
        )
