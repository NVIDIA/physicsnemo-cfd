# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report visuals registry and built-in plot hooks (``physicsnemo.cfd.bench.visualization``)."""

from physicsnemo.cfd.evaluation.reports.registry import (
    get_visual,
    list_visuals,
    register_visual,
)

import physicsnemo.cfd.evaluation.reports.builtin  # noqa: F401 — register built-ins

__all__ = ["register_visual", "get_visual", "list_visuals"]
