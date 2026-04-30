# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-case inference RNG seeding."""

from __future__ import annotations

import numpy as np
import torch

from physicsnemo.cfd.evaluation.common.inference_seed import (
    inference_seed,
    seed_inference_rng,
)


def test_inference_seed_stable_across_calls() -> None:
    a = inference_seed(42, "case_a")
    b = inference_seed(42, "case_a")
    c = inference_seed(42, "case_b")
    assert a == b
    assert a != c


def test_seed_inference_rng_np_and_torch_match_rerun() -> None:
    seed_inference_rng(7, "x")
    ta = np.random.randint(0, 10_000, size=5)
    tb = torch.randn(3)
    seed_inference_rng(7, "x")
    assert np.array_equal(ta, np.random.randint(0, 10_000, size=5))
    assert torch.allclose(tb, torch.randn(3))
