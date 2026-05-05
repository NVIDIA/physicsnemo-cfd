# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
