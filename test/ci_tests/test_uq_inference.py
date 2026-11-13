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

"""Tests for the engine-side UQ plumbing (sampling loop + pooled reducer finalization)."""

from __future__ import annotations

import math

import numpy as np
import pytest

# Registers built-in metrics (including the pooled UQ reducers) into the registry.
import physicsnemo.cfd.evaluation.metrics  # noqa: F401
from physicsnemo.cfd.evaluation.benchmarks.uq_inference import (
    finalize_reducer_metrics,
    finalize_sample_metrics,
    is_reducer_partial_key,
    make_reducer_partial_key,
    run_sampling_inference,
    select_inference_path,
    strip_reducer_partials,
    _Welford,
)
from physicsnemo.cfd.evaluation.datasets.schema import FieldDistribution
from physicsnemo.cfd.postprocessing_tools.metric_registry import get_metric


def test_reducer_partial_key_round_trip_and_strip() -> None:
    """Reducer partial keys round-trip and are stripped from reported per-case rows."""
    key = make_reducer_partial_key("nlpd", "pressure::sum")
    assert key == "_uq::nlpd::pressure::sum"
    assert is_reducer_partial_key(key)
    rows = [{"case_id": "a", "metrics": {"l2_pressure": 0.1, key: 2.0}}]
    strip_reducer_partials(rows)
    assert rows[0]["metrics"] == {"l2_pressure": 0.1}


def test_welford_population_variance() -> None:
    """``_Welford`` yields the streaming mean and population (epistemic) std across passes."""
    w = _Welford()
    for x in (np.array([1.0, 2.0]), np.array([3.0, 2.0]), np.array([5.0, 2.0])):
        w.update(x)
    mean, total, epi, ale = w.result()
    assert np.allclose(mean, [3.0, 2.0])
    assert np.allclose(epi, [np.sqrt(8 / 3), 0.0])  # population variance of {1,3,5}
    assert ale is None and np.allclose(total, epi)


class _EnsembleWrapper:
    SUPPORTS_UQ = True
    UQ_METHOD = "sampling"

    def __init__(self, outs, hetero_std=None):
        self._outs = outs
        self._hetero_std = hetero_std

    def predict_ensemble(self, model_input, n):
        return self._outs[:n]

    def predict(self, model_input):  # pragma: no cover - ensemble path used
        raise AssertionError("ensemble fast-path should be used")

    def decode_outputs(self, raw, case, model_input=None):
        if self._hetero_std is not None:
            return {"pressure": FieldDistribution(mean=raw, std=self._hetero_std)}
        return {"pressure": raw}


def test_run_sampling_inference_epistemic_and_samples() -> None:
    """Ensemble sampling gives the across-member mean, epistemic std, and retained samples."""
    outs = [np.array([1.0, 10.0]), np.array([3.0, 10.0]), np.array([5.0, 10.0])]
    d = run_sampling_inference(
        _EnsembleWrapper(outs),
        None,
        None,
        n=3,
        run_seed=0,
        case_id="c",
        retain_samples=True,
    )
    fd = d["pressure"]
    assert np.allclose(fd.mean, [3.0, 10.0])
    assert np.allclose(fd.epistemic_std, [np.sqrt(8 / 3), 0.0])
    assert np.allclose(fd.std, fd.epistemic_std)  # no aleatoric component
    assert fd.samples.shape == (3, 2)


def test_run_sampling_inference_zero_spread_gives_zero_epistemic() -> None:
    """Identical passes -> zero epistemic std."""
    d = run_sampling_inference(
        _EnsembleWrapper([np.array([2.0, 2.0])] * 4),
        None,
        None,
        n=4,
        run_seed=0,
        case_id="c",
    )
    assert np.allclose(d["pressure"].epistemic_std, 0.0)


def test_run_sampling_inference_law_of_total_variance() -> None:
    """Per-pass aleatoric std combines with across-pass spread via the law of total variance."""
    outs = [np.array([1.0, 10.0]), np.array([3.0, 10.0]), np.array([5.0, 10.0])]
    d = run_sampling_inference(
        _EnsembleWrapper(outs, hetero_std=np.array([2.0, 2.0])),
        None,
        None,
        n=3,
        run_seed=0,
        case_id="c",
    )
    fd = d["pressure"]
    # total_var = var_i(mu_i) + mean_i(sigma_i^2) = 8/3 + 4
    assert np.allclose(fd.aleatoric_std, [2.0, 2.0])
    assert np.allclose(fd.std[0], np.sqrt(8 / 3 + 4))


def test_finalize_reducer_metrics_pools_over_cases() -> None:
    """Reducer finalization pools sufficient statistics over cases (not a mean of case means)."""
    rng = np.random.default_rng(0)

    def _case(n, sigma):
        mu = rng.normal(size=n).astype(np.float32)
        y = (mu + sigma * rng.normal(size=n)).astype(np.float32)
        pred = {
            "pressure": FieldDistribution(
                mean=mu,
                std=np.full(n, sigma, np.float32),
                epistemic_std=np.full(n, sigma * 0.5, np.float32),
            )
        }
        gt = {"pressure": y}
        metrics = {}
        for mn in ("coverage_95", "calibration_zrms", "sharpness_std"):
            m = get_metric(mn, domain="surface")
            for pk, pv in m.partial(gt, pred).items():
                metrics[make_reducer_partial_key(mn, pk)] = pv
        return {"case_id": str(n), "metrics": metrics}

    rows = [_case(300_000, 1.0), _case(100_000, 2.0)]
    summary = finalize_reducer_metrics(rows, "surface")
    assert abs(summary["coverage_95_pressure"] - 0.95) < 0.01
    assert abs(summary["calibration_zrms_pressure"] - 1.0) < 0.02
    # pooled sharpness = (300000*1 + 100000*2) / 400000 = 1.25 (not mean-of-means 1.5)
    assert abs(summary["sharpness_std_pressure"] - 1.25) < 1e-3


# --------------------------------------------------------------------------------------------
# Configured-metric NaN placeholders: deterministic rows report NaN, not omitted keys
# --------------------------------------------------------------------------------------------


def test_finalize_reducer_metrics_emits_nan_for_configured_but_absent() -> None:
    """A deterministic row (no ``_uq::`` partials) still reports configured reducer metrics as NaN.

    The placeholder uses the SAME headline key a populated finalize would (``{metric}_mean``) so
    deterministic and UQ rows share a schema.
    """
    rows = [{"case_id": "a", "metrics": {"l2_pressure": 0.1}}]  # no reducer partials
    # Without the configured names the metric is simply absent (legacy behavior)...
    assert "nlpd_mean" not in finalize_reducer_metrics(rows, "surface")
    # ...with them it is finalized to NaN so the report schema stays consistent.
    summary = finalize_reducer_metrics(rows, "surface", ["nlpd", "coverage_95"])
    assert "nlpd_mean" in summary and math.isnan(summary["nlpd_mean"])
    assert "coverage_95_mean" in summary and math.isnan(summary["coverage_95_mean"])
    # An unknown / non-reducer configured name is ignored (not every metric is a reducer).
    assert "l2_mean" not in finalize_reducer_metrics(rows, "surface", ["l2", "nlpd"])


def test_finalize_sample_metrics_emits_nan_for_configured_but_absent() -> None:
    """A deterministic row (no ``_uqs::`` partials) still reports configured sample metrics as NaN."""
    rows = [{"case_id": "a", "metrics": {"l2_pressure": 0.1}}]
    summary = finalize_sample_metrics(
        rows, "surface", ["sparsification_ause", "drag_uq"]
    )
    assert math.isnan(summary["sparsification_ause_mean"])
    # drag_uq returns a dict -> expands to <metric>_<sub>, all NaN.
    assert math.isnan(summary["drag_uq_epistemic"])
    assert math.isnan(summary["drag_uq_total"])


# --------------------------------------------------------------------------------------------
# Master switch dispatch (run.uq.enabled) + streaming ensemble generator
# --------------------------------------------------------------------------------------------


def test_select_inference_path_master_switch() -> None:
    """``run.uq.enabled`` is the master switch: off -> every wrapper goes deterministic."""
    # UQ enabled: sampling / closed-form wrappers take their UQ path.
    assert (
        select_inference_path(supports_uq=True, uq_method="sampling", uq_enabled=True)
        == "sampling"
    )
    assert (
        select_inference_path(
            supports_uq=True, uq_method="closed_form", uq_enabled=True
        )
        == "closed_form"
    )
    # Master switch OFF: EVERY wrapper (incl. closed-form GP) goes deterministic -> no UQ metrics.
    assert (
        select_inference_path(
            supports_uq=True, uq_method="closed_form", uq_enabled=False
        )
        == "deterministic"
    )
    assert (
        select_inference_path(supports_uq=True, uq_method="sampling", uq_enabled=False)
        == "deterministic"
    )
    # Non-UQ wrapper is always deterministic.
    assert (
        select_inference_path(supports_uq=False, uq_method="none", uq_enabled=True)
        == "deterministic"
    )


class _GeneratorEnsembleWrapper:
    """Ensemble whose ``predict_ensemble`` is a *generator* (one output resident at a time)."""

    SUPPORTS_UQ = True
    UQ_METHOD = "sampling"

    def __init__(self, outs):
        self._outs = outs
        self.live = 0
        self.max_live = 0

    def predict_ensemble(self, model_input, n):
        for o in self._outs:
            self.live += 1
            self.max_live = max(self.max_live, self.live)
            yield o
            self.live -= 1  # engine consumed it before the next is produced

    def predict(self, model_input):  # pragma: no cover
        raise AssertionError("generator ensemble path should be used")

    def decode_outputs(self, raw, case, model_input=None):
        return {"pressure": raw}


def test_run_sampling_inference_consumes_generator_streaming() -> None:
    """A generator ``predict_ensemble`` is consumed lazily (never all members resident at once)."""
    outs = [np.array([1.0, 10.0]), np.array([3.0, 10.0]), np.array([5.0, 10.0])]
    w = _GeneratorEnsembleWrapper(outs)
    d = run_sampling_inference(w, None, None, n=3, run_seed=0, case_id="c")
    assert np.allclose(d["pressure"].mean, [3.0, 10.0])
    assert np.allclose(d["pressure"].epistemic_std, [np.sqrt(8 / 3), 0.0])
    assert w.max_live == 1  # streaming: only one member output alive at any time


class _BudgetEnsembleWrapper:
    """Fixed-size ensemble that honors the budget: yields ``min(n, member_count)`` members."""

    SUPPORTS_UQ = True
    UQ_METHOD = "sampling"

    def __init__(self, members):
        self._members = members
        self.yielded = 0

    def predict_ensemble(self, model_input, n):
        k = min(int(n), len(self._members))
        for m in self._members[:k]:
            self.yielded += 1
            yield m

    def predict(self, model_input):  # pragma: no cover - ensemble path used
        raise AssertionError("ensemble path should be used")

    def decode_outputs(self, raw, case, model_input=None):
        return {"pressure": raw}


def test_ensemble_predict_ensemble_honors_num_samples_budget() -> None:
    """The ensemble caps at its member count but otherwise uses only ``n`` members (honors budget)."""
    members = [np.array([float(i)]) for i in range(5)]
    # n < K: only the first n members contribute.
    w_small = _BudgetEnsembleWrapper(members)
    run_sampling_inference(w_small, None, None, n=2, run_seed=0, case_id="c")
    assert w_small.yielded == 2
    # n > K: cannot fabricate members -> all K used.
    w_big = _BudgetEnsembleWrapper(members)
    run_sampling_inference(w_big, None, None, n=32, run_seed=0, case_id="c")
    assert w_big.yielded == 5


@pytest.fixture
def tensor_statistics():
    """New-backend tests run in the dedicated core-API CI job, and skip on 2.2.0."""
    from physicsnemo.cfd.evaluation.benchmarks.uq_inference import (
        _require_tensor_statistics,
    )

    try:
        return _require_tensor_statistics()
    except RuntimeError:
        pytest.skip("PhysicsNeMo tensor-mode statistics are not available")


@pytest.mark.parametrize("shape", [(7,), (5, 3), ()])
@pytest.mark.parametrize("with_variance", [False, True, "mixed"])
def test_shared_statistics_matches_original(shape, with_variance, tensor_statistics):
    """Shared moments preserve population/total variance, units, and empty behavior."""
    from physicsnemo.cfd.evaluation.benchmarks.uq_inference import _SharedMoments

    reference, shared = _Welford(), _SharedMoments()
    assert shared.result() is None
    rng = np.random.default_rng(12)
    for i in range(9):
        value = 1e6 + rng.normal(size=shape)
        variance = (
            rng.uniform(0.1, 3.0, size=shape)
            if with_variance is True or (with_variance == "mixed" and i % 2)
            else None
        )
        reference.update(value, variance)
        shared.update(value, variance)
    for expected, actual in zip(reference.result(), shared.result()):
        if expected is None:
            assert actual is None
        else:
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("backend", ["numpy", "physicsnemo"])
def test_decoded_units_and_retained_samples(backend, request):
    """Both paths reduce decoded fields and preserve samples and within-member variance."""

    if backend == "physicsnemo":
        request.getfixturevalue("tensor_statistics")

    class DecodingWrapper(_EnsembleWrapper):
        """Decode standardized samples into pressure and pressure uncertainty."""

        def decode_outputs(self, raw, case, model_input=None):
            """Apply distinct affine mean and multiplicative scale transforms."""
            return {
                "pressure": FieldDistribution(
                    mean=raw * 10 + 100, std=np.ones_like(raw) * 20
                )
            }

    values = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    result = run_sampling_inference(
        DecodingWrapper(values),
        None,
        None,
        n=2,
        run_seed=0,
        case_id="units",
        retain_samples=True,
        statistics_backend=backend,
    )["pressure"]
    np.testing.assert_allclose(result.mean, [120.0, 130.0])
    np.testing.assert_allclose(result.epistemic_std, [10.0, 10.0])
    np.testing.assert_allclose(result.aleatoric_std, [20.0, 20.0])
    np.testing.assert_allclose(result.std, np.sqrt([500.0, 500.0]))
    np.testing.assert_allclose(result.samples, [[110.0, 120.0], [130.0, 140.0]])


def test_shared_backend_config_validation():
    """The opt-in is explicit and typos fail before inference."""
    from physicsnemo.cfd.evaluation.config import UQConfig

    assert UQConfig().statistics_backend == "numpy"
    assert (
        UQConfig(statistics_backend="physicsnemo").statistics_backend == "physicsnemo"
    )
    with pytest.raises(ValueError, match="statistics_backend"):
        UQConfig(statistics_backend="typo")
    with pytest.raises(ValueError, match="statistics_backend"):
        run_sampling_inference(
            _EnsembleWrapper([]),
            None,
            None,
            n=1,
            run_seed=0,
            case_id="invalid",
            statistics_backend="typo",
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("retain_samples", [False, True])
def test_shared_statistics_tensor_decoding(device, retain_samples, tensor_statistics):
    """GPU/CPU decoding stays detached and preserves physical units and legacy outputs."""
    import torch

    from physicsnemo.cfd.evaluation.benchmarks.uq_inference import _SharedMoments

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")

    class TensorDecoder(_EnsembleWrapper):
        """Keep decoded means/scales on the model's device."""

        def decode_outputs(self, raw, case, model_input=None):
            """Decode means and a half-precision scale whose square exceeds fp16."""
            return {
                "pressure": FieldDistribution(
                    mean=raw * 10 + 100,
                    std=torch.full_like(raw, 300, dtype=torch.float16),
                )
            }

    values = [
        torch.tensor([1.0, 2.0], device=device, requires_grad=True),
        torch.tensor([3.0, 4.0], device=device, requires_grad=True),
    ]
    accumulator = _SharedMoments()
    for value in values:
        accumulator.update(value)
    assert accumulator._between.mean.device.type == device
    assert not accumulator._between.mean.requires_grad
    result = run_sampling_inference(
        TensorDecoder(values),
        None,
        None,
        n=2,
        run_seed=0,
        case_id="tensor-units",
        retain_samples=retain_samples,
        statistics_backend="physicsnemo",
    )["pressure"]
    assert isinstance(result.mean, np.ndarray)
    assert result.mean.dtype == np.float32
    np.testing.assert_allclose(result.mean, [120.0, 130.0])
    np.testing.assert_allclose(result.epistemic_std, 10)
    np.testing.assert_allclose(result.aleatoric_std, 300)
    np.testing.assert_allclose(result.std, np.sqrt(90100))
    if retain_samples:
        np.testing.assert_allclose(result.samples, [[110.0, 120.0], [130.0, 140.0]])
    else:
        assert result.samples is None


def test_old_physicsnemo_fails_before_sampling(monkeypatch):
    """An unavailable opt-in API must not invoke an expensive model forward."""
    import physicsnemo.metrics.general.ensemble_metrics as metrics

    class LegacyStatistic:
        """Match the constructor contract before tensor mode was added."""

        def __init__(self, input_shape, **kwargs):
            pass

    monkeypatch.setattr(metrics, "Mean", LegacyStatistic)
    monkeypatch.setattr(metrics, "Variance", LegacyStatistic)
    with pytest.raises(RuntimeError, match="statistics_backend='numpy'"):
        run_sampling_inference(
            object(),
            None,
            None,
            n=1,
            run_seed=0,
            case_id="old-core",
            statistics_backend="physicsnemo",
        )


@pytest.mark.parametrize("backend", ["numpy", "physicsnemo"])
def test_single_draw_has_zero_epistemic_variance(backend, request):
    """Population variance is valid for one prediction and keeps its within variance."""
    if backend == "physicsnemo":
        request.getfixturevalue("tensor_statistics")
    result = run_sampling_inference(
        _EnsembleWrapper([np.array([3.0, 4.0])], hetero_std=np.array([2.0, 5.0])),
        None,
        None,
        n=1,
        run_seed=0,
        case_id="single",
        statistics_backend=backend,
    )["pressure"]
    np.testing.assert_array_equal(result.mean, [3.0, 4.0])
    np.testing.assert_array_equal(result.epistemic_std, [0.0, 0.0])
    np.testing.assert_array_equal(result.std, [2.0, 5.0])


def test_sampling_backends_reproduce_seeded_draws(tensor_statistics):
    """The fallback sampling loop uses the same case/pass seeds in both backends."""

    class StochasticWrapper(_EnsembleWrapper):
        """Sample one decoded field per pass without an explicit ensemble."""

        def predict_ensemble(self, model_input, n):
            """Use the repeated-predict path."""
            return None

        def predict(self, model_input):
            """Draw from the seeded NumPy RNG."""
            return np.random.normal(size=7)

    results = [
        run_sampling_inference(
            StochasticWrapper([]),
            None,
            None,
            n=5,
            run_seed=17,
            case_id="stochastic",
            retain_samples=True,
            statistics_backend=backend,
        )["pressure"]
        for backend in ("numpy", "physicsnemo")
    ]
    for field in ("mean", "std", "epistemic_std", "samples"):
        np.testing.assert_allclose(
            getattr(results[0], field), getattr(results[1], field)
        )
    assert np.any(results[1].std > 0)


def test_shared_retained_samples_own_their_storage(tensor_statistics):
    """A decoder may reuse its buffer without corrupting previously retained draws."""
    import torch

    class ReusingWrapper(_EnsembleWrapper):
        """Yield one mutable tensor repeatedly, as an inference buffer might."""

        def predict_ensemble(self, model_input, n):
            """Overwrite storage between draws."""
            value = torch.zeros(2, dtype=torch.float64)
            for i in range(n):
                value.fill_(i)
                yield value

    result = run_sampling_inference(
        ReusingWrapper([]),
        None,
        None,
        n=3,
        run_seed=0,
        case_id="buffer",
        retain_samples=True,
        statistics_backend="physicsnemo",
    )["pressure"]
    np.testing.assert_array_equal(result.samples, [[0, 0], [1, 1], [2, 2]])
    np.testing.assert_allclose(result.mean, [1, 1])
