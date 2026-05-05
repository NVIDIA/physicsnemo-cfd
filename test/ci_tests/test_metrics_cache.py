# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for benchmark metrics cache (no heavy dependencies)."""

from __future__ import annotations

import math
from pathlib import Path

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
