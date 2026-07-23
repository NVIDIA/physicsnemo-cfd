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

"""Tests for the UQ contract (``FieldDistribution``) and pooled UQ reducer metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from physicsnemo.cfd.evaluation.datasets.schema import (
    FieldDistribution,
    as_distribution,
    build_predictive_distribution,
    distribution_mean,
)
from physicsnemo.cfd.postprocessing_tools.metric_registry import (
    is_reducer_metric,
    is_sample_metric,
)
from physicsnemo.cfd.evaluation.metrics.builtin.uq import (
    _ause_curve_area,
    _curve_payload,
    _drag_mean_and_std,
    _make_pointwise_uq_metrics,
    _make_sample_uq_metrics,
    _make_uq_metrics,
    _SampleDragUQ,
    _spearman,
    _trapezoid,
)


def test_build_predictive_distribution_coerces_numpy_to_float32() -> None:
    """``build_predictive_distribution`` coerces mean/std/epistemic arrays to float32."""
    fd = build_predictive_distribution(
        mean=np.ones(4, dtype=np.float64), std=np.full(4, 2.0), epistemic_std=np.ones(4)
    )
    assert fd.mean.dtype == np.float32
    assert fd.std.dtype == np.float32
    assert fd.epistemic_std.dtype == np.float32


def test_as_distribution_passthrough_wrap_and_none() -> None:
    """``as_distribution`` returns distributions as-is, wraps plain arrays, and yields None otherwise."""
    fd = FieldDistribution(mean=np.zeros(3), std=np.ones(3))
    preds = {"pressure": fd, "shear_stress": np.arange(6.0).reshape(3, 2), "z": None}
    assert as_distribution(preds, "pressure") is fd
    wrapped = as_distribution(preds, "shear_stress")
    assert isinstance(wrapped, FieldDistribution) and wrapped.std is None
    assert as_distribution(preds, "z") is None
    assert as_distribution(preds, "missing") is None


def test_distribution_mean_unwraps() -> None:
    """``distribution_mean`` returns the mean array for both distributions and plain arrays."""
    fd = FieldDistribution(mean=np.arange(3.0), std=np.ones(3))
    assert distribution_mean(fd) is fd.mean
    arr = np.zeros(3)
    assert distribution_mean(arr) is arr


def _run_reducer(metric, cases):
    """Mimic the engine: sum per-case ``partial`` stats, then ``finalize`` once."""
    summed: dict[str, float] = {}
    for gt, pred in cases:
        for key, val in metric.partial(gt, pred).items():
            summed[key] = summed.get(key, 0.0) + val
    return metric.finalize(summed)


def _gaussian_case(rng, n, sigma, epi=None):
    mu = rng.normal(size=n).astype(np.float32)
    y = (mu + sigma * rng.normal(size=n)).astype(np.float32)
    fd = FieldDistribution(
        mean=mu,
        std=np.full(n, sigma, np.float32),
        epistemic_std=None if epi is None else np.full(n, epi, np.float32),
    )
    return {"pressure": y}, {"pressure": fd}


def test_uq_metrics_are_registered_as_reducers() -> None:
    """The pooled UQ metrics are all reducer metrics."""
    metrics = _make_uq_metrics()
    assert set(metrics) >= {
        "nlpd",
        "nlpd_epistemic",
        "calibration_zrms",
        "coverage_95",
        "sharpness_std",
        "sharpness_epistemic_std",
    }
    for m in metrics.values():
        assert is_reducer_metric(m)


def test_pooled_calibration_on_synthetic_gaussian() -> None:
    """A perfectly calibrated Gaussian gives cov95≈0.95, zrms≈1, and the analytic NLPD."""
    rng = np.random.default_rng(0)
    metrics = _make_uq_metrics()
    n, sigma = 400_000, 1.0
    cases = [_gaussian_case(rng, n, sigma)]
    assert abs(_run_reducer(metrics["coverage_95"], cases)["pressure"] - 0.95) < 0.01
    assert (
        abs(_run_reducer(metrics["calibration_zrms"], cases)["pressure"] - 1.0) < 0.02
    )
    nlpd = _run_reducer(metrics["nlpd"], cases)["pressure"]
    expected = 0.5 * (math.log(2 * math.pi) + math.log(sigma**2) + 1.0)
    assert abs(nlpd - expected) < 0.02


def test_pooling_is_global_not_mean_of_case_means() -> None:
    """Sharpness over two cases with different point counts pools by point, not by case."""
    rng = np.random.default_rng(1)
    metrics = _make_uq_metrics()
    n1, n2, s1, s2 = 300_000, 100_000, 1.0, 3.0
    cases = [_gaussian_case(rng, n1, s1), _gaussian_case(rng, n2, s2)]
    pooled = _run_reducer(metrics["sharpness_std"], cases)["pressure"]
    expected_pooled = (n1 * s1 + n2 * s2) / (n1 + n2)
    mean_of_means = (s1 + s2) / 2.0
    assert abs(pooled - expected_pooled) < 1e-3
    assert abs(pooled - mean_of_means) > 0.05


def test_deterministic_prediction_yields_nan() -> None:
    """A deterministic prediction (no std) yields NaN NLPD under the headline ``mean`` sub-key."""
    metrics = _make_uq_metrics()
    cases = [
        (
            {"pressure": np.zeros(16, np.float32)},
            {"pressure": FieldDistribution(mean=np.zeros(16, np.float32))},
        )
    ]
    # No channels contribute -> finalize({}) returns the {"mean": NaN} headline (consistent schema).
    assert math.isnan(_run_reducer(metrics["nlpd"], cases)["mean"])


def test_epistemic_metric_requires_epistemic_std() -> None:
    """Epistemic metrics need an epistemic_std: absent -> NaN headline; present -> finite."""
    rng = np.random.default_rng(2)
    metrics = _make_uq_metrics()
    # total std present but epistemic_std absent -> epistemic metric is NaN (headline "mean").
    cases_no_epi = [_gaussian_case(rng, 1000, 1.0)]
    assert math.isnan(_run_reducer(metrics["nlpd_epistemic"], cases_no_epi)["mean"])
    # epistemic_std present -> finite
    cases_epi = [_gaussian_case(rng, 1000, 1.0, epi=0.5)]
    res = _run_reducer(metrics["sharpness_epistemic_std"], cases_epi)
    assert abs(res["pressure"] - 0.5) < 1e-3


def test_vector_field_splits_into_components() -> None:
    """A 3-vector field expands into per-component channels plus the headline mean."""
    rng = np.random.default_rng(3)
    metrics = _make_uq_metrics()
    n, sigma, epi = 50_000, 1.0, 0.5
    mu = rng.normal(size=(n, 3)).astype(np.float32)
    y = (mu + sigma * rng.normal(size=(n, 3))).astype(np.float32)
    fd = FieldDistribution(
        mean=mu,
        std=np.full((n, 3), sigma, np.float32),
        epistemic_std=np.full((n, 3), epi, np.float32),
    )
    res = _run_reducer(
        metrics["nlpd_epistemic"], [({"shear_stress": y}, {"shear_stress": fd})]
    )
    assert set(res) == {"shear_stress_x", "shear_stress_y", "shear_stress_z", "mean"}


def test_partial_outputs_are_additive_scalars() -> None:
    """Reducer ``partial`` returns a flat dict of floats so it caches like per-case scalars."""
    rng = np.random.default_rng(4)
    metrics = _make_uq_metrics()
    gt, pred = _gaussian_case(rng, 100, 1.0)
    part = metrics["coverage_95"].partial(gt, pred)
    assert part and all(isinstance(v, float) for v in part.values())


# --------------------------------------------------------------------------------------------
# Pointwise per-point ranking quality: Spearman(|error|, uncertainty)
# --------------------------------------------------------------------------------------------


def test_spearman_high_when_uncertainty_tracks_error() -> None:
    """Spearman ≈ 1 when the per-point std equals the noise scale that generated the error."""
    rng = np.random.default_rng(5)
    spear = _make_pointwise_uq_metrics()["uncertainty_error_spearman"]
    n = 20_000
    mu = rng.normal(size=n).astype(np.float32)
    sig = rng.uniform(0.2, 3.0, size=n).astype(np.float32)
    # Make |error| a monotonic function of sig so ranks align almost perfectly.
    y = (mu + sig * 3.0).astype(np.float32)
    informative = spear(
        {"pressure": y}, {"pressure": FieldDistribution(mean=mu, std=sig)}
    )
    assert informative["pressure"] > 0.99

    random_sig = rng.uniform(0.2, 3.0, size=n).astype(np.float32)
    uninformative = spear(
        {"pressure": y}, {"pressure": FieldDistribution(mean=mu, std=random_sig)}
    )
    assert abs(uninformative["pressure"]) < 0.1


def test_spearman_tie_aware_and_constant_input() -> None:
    """Ties use *average* ranks and a constant input is undefined (NaN), not spuriously 1.0."""
    from scipy.stats import spearmanr

    # Increasing error paired with completely CONSTANT uncertainty: rank correlation is undefined
    # (zero rank variance). The old ordinal-rank implementation wrongly reported rho = 1.0.
    err = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    const_unc = np.full(5, 0.7)
    assert math.isnan(_spearman(err, const_unc))
    assert math.isnan(_spearman(const_unc, err))
    # Tie-aware: half the uncertainties tie but the trend is still monotone -> strongly positive
    # (average ranks give ~0.949; ordinal ranks would spuriously read higher/inconsistent).
    tied_unc = np.array([0.1, 0.1, 0.2, 0.2, 0.3])
    mono_err = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    assert _spearman(mono_err, tied_unc) > 0.9
    assert (
        abs(
            _spearman(mono_err, tied_unc)
            - float(spearmanr(mono_err, tied_unc).correlation)
        )
        < 1e-12
    )
    # Matches scipy (which we delegate to) on a small tied example.
    a = np.array([1.0, 2.0, 2.0, 3.0])
    b = np.array([10.0, 20.0, 20.0, 40.0])
    assert abs(_spearman(a, b) - float(spearmanr(a, b).correlation)) < 1e-12
    # Degenerate sizes -> NaN.
    assert math.isnan(_spearman(np.array([1.0]), np.array([1.0])))
    assert math.isnan(_spearman(np.array([1.0, 2.0]), np.array([1.0])))


def test_trapezoid_available_and_ause_uses_it() -> None:
    """``_trapezoid`` resolves (np.trapz was removed in NumPy 2.4) and AUSE integrates cleanly."""
    assert callable(_trapezoid)
    assert _trapezoid(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.5, 1.0])) == 0.5
    # Perfectly-ranked errors -> AUSE ~ 0 (no AttributeError from a missing np.trapz).
    err = np.array([0.2, 0.5, 1.0, 2.0, 4.0])
    assert abs(_ause_curve_area(err, err)) < 1e-9


def test_spearman_deterministic_and_epistemic_variants() -> None:
    """Deterministic (no std) and epistemic-without-epistemic-std both give a NaN Spearman mean."""
    spear = _make_pointwise_uq_metrics()["uncertainty_error_spearman"]
    spear_epi = _make_pointwise_uq_metrics()["uncertainty_error_spearman_epistemic"]
    n = 4000
    mu = np.zeros(n, np.float32)
    y = np.random.default_rng(6).normal(size=n).astype(np.float32)
    assert math.isnan(
        spear({"pressure": y}, {"pressure": FieldDistribution(mean=mu)})["mean"]
    )
    fd_total = FieldDistribution(mean=mu, std=np.ones(n, np.float32))
    assert math.isnan(spear_epi({"pressure": y}, {"pressure": fd_total})["mean"])


# --------------------------------------------------------------------------------------------
# Sample-wise sparsification / AUSE (sample metric: one number per geometry)
# --------------------------------------------------------------------------------------------


def _collect_samples(metric, cases):
    """Mimic the engine: gather per-case ``partial`` scalars into lists, then finalize_samples."""
    collected: dict[str, list[float]] = {}
    for gt, pred in cases:
        for key, val in metric.partial(gt, pred).items():
            collected.setdefault(key, []).append(val)
    return metric.finalize_samples(collected)


def test_sample_ause_is_registered_as_sample_metric() -> None:
    """The sample-wise AUSE metrics are sample metrics (finalize_samples, not finalize)."""
    for m in _make_sample_uq_metrics().values():
        assert is_sample_metric(m)
        assert not is_reducer_metric(
            m
        )  # sample metrics use finalize_samples, not finalize


def test_sample_ause_informative_below_random_over_geometries() -> None:
    """Rank geometries by mean uncertainty: informative uncertainty gives lower AUSE than random."""
    rng = np.random.default_rng(7)
    ause = _make_sample_uq_metrics()["sparsification_ause"]

    def _geom(scale, rank_sig):
        n = 2000
        mu = np.zeros(n, np.float32)
        y = (scale * rng.normal(size=n)).astype(np.float32)  # error grows with `scale`
        fd = FieldDistribution(mean=mu, std=np.full(n, rank_sig, np.float32))
        return {"pressure": y}, {"pressure": fd}

    scales = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
    # Informative: per-geometry uncertainty == its error scale (ranks geometries correctly).
    informative = _collect_samples(ause, [_geom(s, s) for s in scales])
    # Uninformative: uncertainty unrelated to error scale.
    shuffled = list(scales)
    rng.shuffle(shuffled)
    random_rank = _collect_samples(
        ause, [_geom(s, r) for s, r in zip(scales, shuffled)]
    )
    assert informative["pressure"] <= random_rank["pressure"] + 1e-9
    assert abs(informative["pressure"]) < 1e-6  # perfect ranking -> AUSE ~ 0
    # Spearman companion: informative ranking -> rho ~ 1, and never worse than random ranking.
    assert informative["pressure_spearman"] >= random_rank["pressure_spearman"] - 1e-9
    assert informative["pressure_spearman"] > 0.99


def test_sample_ause_needs_two_geometries() -> None:
    """AUSE needs >= 2 geometries; a single geometry yields NaN for that channel."""
    ause = _make_sample_uq_metrics()["sparsification_ause"]
    n = 100
    fd = FieldDistribution(mean=np.zeros(n, np.float32), std=np.ones(n, np.float32))
    res = _collect_samples(
        ause, [({"pressure": np.ones(n, np.float32)}, {"pressure": fd})]
    )
    assert math.isnan(res["pressure"])


def test_sample_ause_curves_shape() -> None:
    """``curves`` returns a per-channel payload with aligned fraction/curve arrays and AUSE."""
    ause = _make_sample_uq_metrics()["sparsification_ause"]
    collected = {
        "pressure::err": [0.2, 0.5, 1.0, 2.0],
        "pressure::unc": [0.2, 0.5, 1.0, 2.0],  # perfect ranking
    }
    curves = ause.curves(collected)
    assert "pressure" in curves
    c = curves["pressure"]
    assert c["n"] == 4 and len(c["fractions"]) == 4
    assert len(c["by_uncertainty"]) == 4 and len(c["oracle"]) == 4
    assert abs(c["ause"]) < 1e-9  # uncertainty == error order -> curve == oracle


# --------------------------------------------------------------------------------------------
# Sample-wise drag sparsification via linear UQ propagation into Cd (drag_uq)
# --------------------------------------------------------------------------------------------


def test_drag_mean_and_std_matches_manual_propagation() -> None:
    """``_drag_mean_and_std`` reproduces the force integral and the diagonal variance sum."""
    rng = np.random.default_rng(11)
    n = 500
    normals = rng.normal(size=(n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    area = rng.uniform(0.1, 1.0, size=n)
    direction = np.array([1.0, 0.0, 0.0])
    p = rng.normal(size=n)
    wss = rng.normal(size=(n, 3))
    p_std = rng.uniform(0.01, 0.1, size=n)
    wss_std = rng.uniform(0.01, 0.1, size=(n, 3))
    drag, std = _drag_mean_and_std(
        normals, area, direction, p, wss, p_std, wss_std, 1.0
    )
    c_p = float(np.sum((normals @ direction) * area * p))
    c_f = -float(np.sum((wss @ direction) * area))
    assert abs(drag - (c_p + c_f)) < 1e-9
    w_p = (normals @ direction) * area
    w_tau = -area[:, None] * direction[None, :]
    var = float(np.sum(w_p**2 * p_std**2)) + float(np.sum(w_tau**2 * wss_std**2))
    assert abs(std - math.sqrt(var)) < 1e-9
    # zero std -> zero drag std (deterministic mean, no uncertainty).
    _, zero_std = _drag_mean_and_std(
        normals, area, direction, p, wss, np.zeros(n), np.zeros((n, 3)), 1.0
    )
    assert zero_std == 0.0


def test_drag_uq_finalize_and_curves_rank_geometries() -> None:
    """Informative drag std -> AUSE ~ 0; anti-correlated std -> larger AUSE; both series present."""
    drag_uq = _SampleDragUQ()
    err = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
    collected = {
        "abs_err": err,
        "epi_std": err,  # perfectly ranks the drag error
        "total_std": err[::-1],  # anti-correlated
    }
    fin = drag_uq.finalize_samples(collected)
    assert set(fin) == {
        "epistemic",
        "total",
        "epistemic_spearman",
        "total_spearman",
    }
    assert abs(fin["epistemic"]) < 1e-9
    assert fin["total"] > fin["epistemic"]
    # Trend-alignment companion: perfect rank -> rho=+1, anti-correlated -> rho=-1.
    assert abs(fin["epistemic_spearman"] - 1.0) < 1e-9
    assert abs(fin["total_spearman"] + 1.0) < 1e-9
    curves = drag_uq.curves(collected)
    assert set(curves) == {"epistemic", "total"}
    assert curves["epistemic"]["n"] == 6


def test_drag_uq_partial_skips_deterministic() -> None:
    """No std on the prediction (deterministic wrapper) -> ``partial`` is a no-op."""
    drag_uq = _SampleDragUQ()
    preds = {
        "pressure": FieldDistribution(mean=np.zeros(8, np.float32)),
        "shear_stress": FieldDistribution(mean=np.zeros((8, 3), np.float32)),
    }
    # Also confirms ``partial`` accepts the same per-run overrides as the deterministic drag metric.
    assert (
        drag_uq.partial(
            {}, preds, output=object(), coeff=2.5, drag_direction=[1.0, 0.0, 0.0]
        )
        == {}
    )


def test_curve_payload_none_for_single_sample() -> None:
    """The sparsification curve payload is None for a single geometry."""
    assert _curve_payload(np.array([1.0]), np.array([1.0])) is None
