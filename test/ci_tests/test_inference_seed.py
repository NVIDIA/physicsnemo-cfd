# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-case inference RNG seeding."""

from __future__ import annotations

import random
from collections.abc import Iterator

import numpy as np
import pytest
import torch

from physicsnemo.cfd.evaluation.common.inference_seed import (
    inference_seed,
    seed_inference_rng,
)


@pytest.fixture(autouse=True)
def _restore_global_rng_state() -> Iterator[None]:
    """``seed_inference_rng`` mutates global RNGs; restore so other tests stay order-independent."""
    np_state = np.random.get_state()
    py_state = random.getstate()
    torch_cpu = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    yield
    np.random.set_state(np_state)
    random.setstate(py_state)
    torch.set_rng_state(torch_cpu)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


def test_inference_seed_stable_across_calls() -> None:
    """``inference_seed`` is deterministic per ``(base, case_id)`` and varies across cases."""
    a = inference_seed(42, "case_a")
    b = inference_seed(42, "case_a")
    c = inference_seed(42, "case_b")
    assert a == b
    assert a != c


def test_seed_inference_rng_np_and_torch_match_rerun() -> None:
    """Re-seeding ``numpy``/``torch`` with the same key reproduces the same draws."""
    seed_inference_rng(7, "x")
    ta = np.random.randint(0, 10_000, size=5)
    tb = torch.randn(3)
    seed_inference_rng(7, "x")
    assert np.array_equal(ta, np.random.randint(0, 10_000, size=5))
    assert torch.allclose(tb, torch.randn(3))
