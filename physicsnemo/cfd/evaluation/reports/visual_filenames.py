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

"""Shared filename patterns for visuals written under ``<output>/visuals/``."""

from __future__ import annotations


def sanitize_visual_fragment(s: str) -> str:
    """Safe substring for PNG/VTK under ``visuals/`` (alphanumeric, ``-``, ``_``, ``.``; else ``_``).

    Mirrors the logic used for aggregate-volume stems so benchmarks across plot types stay consistent.
    """
    return "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(s)
    )


def join_benchmark_visual_segments(*segments: str) -> str:
    """Join sanitized fragments with underscores (``model_dataset_case_…`` convention)."""
    return "_".join(sanitize_visual_fragment(s) for s in segments)


def benchmark_visual_png(*segments: str) -> str:
    """basename for ``.png`` under ``visuals/`` (suffix included)."""
    return join_benchmark_visual_segments(*segments) + ".png"
