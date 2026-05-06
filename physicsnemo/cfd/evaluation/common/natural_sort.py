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
