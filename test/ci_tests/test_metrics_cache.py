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

"""Tests for benchmark metrics caching and engine compatibility."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import Mock

import pytest

from physicsnemo.cfd.evaluation.benchmarks.metrics_cache import (
    case_id_cache_filename,
    metrics_cache_fingerprint,
    metrics_cache_file_path,
    metrics_from_cache_json,
    output_config_to_fingerprint_dict,
    read_metrics_cache,
    resolve_metrics_cache_root,
    write_metrics_cache,
)
from physicsnemo.cfd.evaluation.config import Config, OutputConfig


def test_resolve_metrics_cache_root_disabled() -> None:
    """Disabled cache always resolves to ``None``."""
    assert (
        resolve_metrics_cache_root(enabled=False, path="", output_dir="/tmp/out")
        is None
    )


def test_resolve_metrics_cache_default_dir(tmp_path: Path) -> None:
    """Without an explicit path the cache lives at ``<output_dir>/.metrics_cache``."""
    root = resolve_metrics_cache_root(
        enabled=True, path="", output_dir=str(tmp_path / "results")
    )
    assert root is not None
    assert root.name == ".metrics_cache"
    assert root.parent.name == "results"


def test_resolve_metrics_cache_explicit_path(tmp_path: Path) -> None:
    """An explicit path is honored verbatim (resolved to absolute)."""
    p = tmp_path / "my_cache"
    root = resolve_metrics_cache_root(
        enabled=True, path=str(p), output_dir=str(tmp_path)
    )
    assert root == p.resolve()


def test_fingerprint_changes_with_checkpoint() -> None:
    """Fingerprint changes when the checkpoint path changes; identical inputs reproduce digests."""
    base = dict(
        model_name="m",
        model_checkpoint="a.pt",
        model_stats_path="",
        model_kwargs={},
        model_inference_domain=None,
        dataset_name="d",
        dataset_root="/data",
        dataset_kwargs_resolved={},
        output_dict=output_config_to_fingerprint_dict(OutputConfig()),
        metric_specs=[("l2_pressure", {})],
    )
    fp1 = metrics_cache_fingerprint(**base)
    fp2 = metrics_cache_fingerprint(**{**base, "model_checkpoint": "b.pt"})
    assert fp1 != fp2
    fp_same = metrics_cache_fingerprint(**base)
    assert fp1 == fp_same


def test_fingerprint_canonical_dataset_root() -> None:
    """Equivalent paths must yield the same digest (stable cache lookups)."""
    out = output_config_to_fingerprint_dict(OutputConfig())
    base = dict(
        model_name="m",
        model_checkpoint="/tmp/a.pt",
        model_stats_path="/tmp/st.json",
        model_kwargs={},
        model_inference_domain=None,
        dataset_name="d",
        dataset_root="/tmp/../tmp/dataset",
        dataset_kwargs_resolved={},
        output_dict=out,
        metric_specs=[("l2_pressure", {})],
    )
    fp_norm = metrics_cache_fingerprint(**{**base, "dataset_root": "/tmp/dataset"})
    fp_canon = metrics_cache_fingerprint(**base)
    assert fp_norm == fp_canon


def test_fingerprint_changes_with_run_seed() -> None:
    """Adding ``run_seed`` to the fingerprint changes the digest."""
    out = output_config_to_fingerprint_dict(OutputConfig())
    base = dict(
        model_name="m",
        model_checkpoint="c.pt",
        model_stats_path="",
        model_kwargs={},
        model_inference_domain="surface",
        dataset_name="d",
        dataset_root="/r",
        dataset_kwargs_resolved={},
        output_dict=out,
        metric_specs=[("l2_pressure", {})],
    )
    fp1 = metrics_cache_fingerprint(**base)
    fp2 = metrics_cache_fingerprint(**{**base, "run_seed": 123})
    assert fp1 != fp2


def test_fingerprint_numpy_model_kwargs_stable() -> None:
    """Numpy scalars in ``model_kwargs`` produce the same digest as native Python ints."""
    np = pytest.importorskip("numpy")
    out = output_config_to_fingerprint_dict(OutputConfig())
    base = dict(
        model_name="m",
        model_checkpoint="c.pt",
        model_stats_path="",
        model_kwargs={"batch_resolution": np.int64(60000)},
        model_inference_domain="surface",
        dataset_name="d",
        dataset_root="/r",
        dataset_kwargs_resolved={},
        output_dict=out,
        metric_specs=[("l2_pressure", {})],
    )
    fp_np = metrics_cache_fingerprint(**base)
    fp_py = metrics_cache_fingerprint(
        **{**base, "model_kwargs": {"batch_resolution": 60000}}
    )
    assert fp_np == fp_py


def test_fingerprint_metric_spec_kwargs_order() -> None:
    """Metric-spec kwargs are sorted before hashing so ordering does not affect the digest."""
    out = output_config_to_fingerprint_dict(OutputConfig())
    base = dict(
        model_name="m",
        model_checkpoint="c.pt",
        model_stats_path="",
        model_kwargs={},
        model_inference_domain="surface",
        dataset_name="d",
        dataset_root="/r",
        dataset_kwargs_resolved={"xyz": 1},
        output_dict=out,
        metric_specs=[("l2_pressure", {"foo": 1, "bar": 2})],
    )
    fp_a = metrics_cache_fingerprint(**base)
    fp_b = metrics_cache_fingerprint(
        **{**base, "metric_specs": [("l2_pressure", {"bar": 2, "foo": 1})]}
    )
    assert fp_a == fp_b


def test_write_read_roundtrip(tmp_path: Path) -> None:
    """Writing then reading a cache file round-trips fingerprints, metrics, and NaNs."""
    fp = "a" * 64
    path = metrics_cache_file_path(tmp_path / "root", fp, "run_1")
    write_metrics_cache(
        path,
        fingerprint=fp,
        model="geotransolver_surface",
        dataset="drivaerml",
        case_id="run_1",
        case_metrics={"l2_pressure": 0.5, "l2_shear_stress": float("nan")},
        metric_dtype="float64",
        comparison_mesh_path="/abs/mesh.vtp",
    )
    blob = read_metrics_cache(path)
    assert blob is not None
    assert blob["fingerprint"] == fp
    m = metrics_from_cache_json(blob["metrics"])
    assert m is not None
    assert m["l2_pressure"] == 0.5
    assert math.isnan(m["l2_shear_stress"])


def test_read_missing_returns_none(tmp_path: Path) -> None:
    """Reading a non-existent cache file returns ``None``."""
    assert read_metrics_cache(tmp_path / "nope.json") is None


def test_read_bad_json_returns_none(tmp_path: Path) -> None:
    """Corrupted JSON in the cache file returns ``None`` rather than raising."""
    p = tmp_path / "bad.json"
    p.write_text("not json {", encoding="utf-8")
    assert read_metrics_cache(p) is None


@pytest.mark.parametrize(
    "cid, expected",
    [
        ("run_1", "run_1.json"),
        ("a/b", "a_b.json"),
    ],
)
def test_case_id_cache_filename(cid: str, expected: str) -> None:
    """``case_id`` strings are sanitized into safe ``.json`` filenames."""
    assert case_id_cache_filename(cid) == expected


def test_config_from_dict_metrics_cache() -> None:
    """``run.metrics_cache`` settings load through :meth:`Config.from_dict`."""
    cfg = Config.from_dict(
        {
            "run": {
                "metrics_cache": {"enabled": True, "path": "/tmp/mc"},
            },
        }
    )
    assert cfg.run.metrics_cache.enabled is True
    assert cfg.run.metrics_cache.path == "/tmp/mc"


def test_config_metrics_cache_enabled_string_false() -> None:
    """``metrics_cache.enabled`` accepts string ``"false"`` as Boolean false (Hydra-safe)."""
    cfg = Config.from_dict(
        {"run": {"metrics_cache": {"enabled": "false", "path": ""}}},
    )
    assert cfg.run.metrics_cache.enabled is False


@pytest.mark.parametrize("uq_enabled", [False, True])
@pytest.mark.parametrize("backend", ["numpy", "physicsnemo"])
def test_engine_reuses_legacy_cache_only_for_default_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, uq_enabled: bool, backend: str
) -> None:
    """Old cache entries skip inference unless a different accumulator is selected."""
    from physicsnemo.cfd.evaluation.benchmarks import engine
    from physicsnemo.cfd.evaluation.config import (
        DatasetConfig,
        MetricsCacheConfig,
        ModelConfig,
        RunConfig,
        UQConfig,
    )

    model = ModelConfig(name="cache_test_model", inference_domain="surface")
    dataset = DatasetConfig(name="cache_test_dataset", root=str(tmp_path / "data"))
    output = OutputConfig()
    run = RunConfig(
        device="cpu",
        metrics_cache=MetricsCacheConfig(enabled=True, path=str(tmp_path / "cache")),
        uq=UQConfig(enabled=uq_enabled, statistics_backend=backend),
    )
    metrics = [("l2_pressure", {})]
    # This is the fingerprint input before statistics_backend was introduced.
    legacy_fingerprint = metrics_cache_fingerprint(
        model_name=model.name,
        model_checkpoint="",
        model_stats_path="",
        model_kwargs={},
        model_inference_domain="surface",
        dataset_name=dataset.name,
        dataset_root=dataset.root,
        dataset_kwargs_resolved={},
        output_dict=output_config_to_fingerprint_dict(output),
        metric_specs=metrics,
        run_seed=run.seed,
        run_uq={"enabled": uq_enabled, "num_samples": 32, "retain_samples": False},
    )
    cache_file = metrics_cache_file_path(
        Path(run.metrics_cache.path), legacy_fingerprint, "case_1"
    )
    write_metrics_cache(
        cache_file,
        fingerprint=legacy_fingerprint,
        model=model.name,
        dataset=dataset.name,
        case_id="case_1",
        case_metrics={"l2_pressure": 0.125},
    )
    adapter_class = Mock()
    adapter_class.inference_domain_from_kwargs.return_value = "surface"
    wrapper_class = Mock(side_effect=AssertionError("Inference would run"))
    monkeypatch.setattr(engine, "get_adapter", lambda name: adapter_class)
    monkeypatch.setattr(engine, "get_model_wrapper", lambda name: wrapper_class)
    monkeypatch.setattr(
        engine, "resolve_model_assets", lambda *args: ("", "", None, {})
    )
    read_cache = Mock(wraps=read_metrics_cache)
    monkeypatch.setattr(engine, "read_metrics_cache", read_cache)

    kwargs = dict(
        model_config=model,
        dataset_config=dataset,
        metric_names=metrics,
        device="cpu",
        output_dir=str(tmp_path / "results"),
        case_ids=["case_1"],
        output_config=output,
        run_config=run,
    )
    if backend == "numpy":
        result, _ = engine._run_single(**kwargs)
        assert result["metrics"]["l2_pressure"] == 0.125
        read_cache.assert_called_once_with(cache_file)
        wrapper_class.assert_not_called()
        adapter_class.return_value.load_case.assert_not_called()
    else:
        with pytest.raises(AssertionError, match="Inference would run"):
            engine._run_single(**kwargs)
        read_cache.assert_called_once()
        assert read_cache.call_args.args[0] != cache_file
        wrapper_class.assert_called_once()
