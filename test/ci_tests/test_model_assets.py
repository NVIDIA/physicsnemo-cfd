# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluation.assets (local Package, resolve_model_assets, cache fingerprint)."""

from __future__ import annotations

from pathlib import Path

import pytest

from physicsnemo.cfd.evaluation.assets import (
    AssetSpec,
    Package,
    clear_default_assets_for_testing,
    register_default_asset,
    resolve_model_assets,
)
from physicsnemo.cfd.evaluation.benchmarks.metrics_cache import metrics_cache_fingerprint
from physicsnemo.cfd.evaluation.config import ModelConfig
from physicsnemo.cfd.evaluation.inference.wrappers.surface_baseline import SurfaceBaselineWrapper


def test_package_resolve_local_file(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    root.mkdir()
    f = root / "global_stats.json"
    f.write_text("{}")
    pkg = Package(str(root))
    assert Path(pkg.resolve("global_stats.json")) == f.resolve()


def test_package_resolve_local_missing(tmp_path: Path) -> None:
    pkg = Package(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        pkg.resolve("missing.pt")


class _TrainedStub:
    REQUIRES_REMOTE_ASSETS = True


def test_resolve_model_assets_explicit_paths() -> None:
    cfg = ModelConfig(
        name="fignet",
        checkpoint="/abs/ck.pt",
        stats_path="/abs/stats.json",
    )
    ck, st, aid = resolve_model_assets(cfg, _TrainedStub)
    assert ck == "/abs/ck.pt" and st == "/abs/stats.json" and aid is None


def test_resolve_model_assets_missing_raises() -> None:
    clear_default_assets_for_testing()
    try:
        cfg = ModelConfig(name="fignet", checkpoint="", stats_path="")
        with pytest.raises(ValueError, match="checkpoint"):
            resolve_model_assets(cfg, _TrainedStub)
    finally:
        clear_default_assets_for_testing()


def test_resolve_model_assets_baseline_no_paths() -> None:
    cfg = ModelConfig(name="surface_baseline", checkpoint="", stats_path="")
    ck, st, aid = resolve_model_assets(cfg, SurfaceBaselineWrapper)
    assert ck == "" and st == "" and aid is None


def test_resolve_model_assets_via_registry(tmp_path: Path) -> None:
    clear_default_assets_for_testing()
    root = tmp_path / "hub_sim"
    root.mkdir()
    (root / "w.pt").write_text("x")
    (root / "stats.json").write_text("{}")
    try:
        register_default_asset(
            "_ci_test_model",
            AssetSpec(
                package_root=str(root),
                checkpoint_relpath="w.pt",
                stats_relpath="stats.json",
            ),
        )
        cfg = ModelConfig(name="_ci_test_model", checkpoint="", stats_path="")
        ck, st, aid = resolve_model_assets(cfg, _TrainedStub)
        assert aid == f"package:{root}|w.pt|stats.json"
        assert Path(ck).name == "w.pt"
        assert Path(st).name == "stats.json"
    finally:
        clear_default_assets_for_testing()


def test_metrics_cache_fingerprint_asset_identity_changes_digest() -> None:
    base = dict(
        model_name="m",
        model_checkpoint="",
        model_stats_path="",
        model_kwargs={},
        model_inference_domain="surface",
        model_asset_identity="package:hf://a/b@main|ck|st",
        dataset_name="d",
        dataset_root="/r",
        dataset_split=None,
        dataset_kwargs_resolved={},
        output_dict={},
        metric_specs=[("l2_pressure", {})],
    )
    fp1 = metrics_cache_fingerprint(**base)
    fp2 = metrics_cache_fingerprint(**{**base, "model_asset_identity": "package:hf://a/b@main|ck2|st"})
    assert fp1 != fp2

    fp_explicit = metrics_cache_fingerprint(
        **{**base, "model_asset_identity": None, "model_checkpoint": "a.pt", "model_stats_path": "b.json"},
    )
    assert fp_explicit != fp1
