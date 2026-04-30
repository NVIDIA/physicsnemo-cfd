# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alphanatural ("natural") sort for identifiers such as ``run_2``, ``run_10``."""

from __future__ import annotations

import re
from typing import Any, Iterable, TypeVar

_T = TypeVar("_T")


def natural_sort_key(value: Any) -> tuple[Any, ...]:
    """Key for sorting strings so numeric sub-fields order numerically (``run_2`` before ``run_10``).

    Digit runs become ``int``; other runs are folded for case-insensitive ASCII ordering.
    """
    s = "" if value is None else str(value)
    parts: list[Any] = []
    for chunk in re.split(r"(\d+)", s):
        if chunk == "":
            continue
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            parts.append(chunk.casefold())
    return tuple(parts)


def natural_sorted(sequence: Iterable[_T]) -> list[_T]:
    """Like :func:`sorted` but ordered by :func:`natural_sort_key`."""
    return sorted(sequence, key=natural_sort_key)
