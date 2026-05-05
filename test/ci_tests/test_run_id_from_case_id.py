# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`run_id_from_case_id` (datapipe STL run index)."""

from __future__ import annotations

import pytest

from physicsnemo.cfd.evaluation.models.common_wrapper_utils.vtk_datapipe_io import (
    run_id_from_case_id,
)


def test_run_id_from_case_id_accepts_run_prefix_and_integers() -> None:
    """``run_<n>`` and bare integers (with surrounding whitespace) parse to the integer index."""
    assert run_id_from_case_id("run_1") == 1
    assert run_id_from_case_id("run_42") == 42
    assert run_id_from_case_id("7") == 7
    assert run_id_from_case_id("  3 ") == 3


@pytest.mark.parametrize(
    "bad",
    [
        "",
        " typo",
        "rub_5",
        "run_",
        "run_x",
        "1.5",
        "nan",
    ],
)
def test_run_id_from_case_id_rejects_unknown_forms(bad: str) -> None:
    """Malformed case-id strings raise ``ValueError`` rather than silently returning ``None``."""
    with pytest.raises(ValueError):
        run_id_from_case_id(bad)
