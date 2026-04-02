# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-export bench metric registry for ``physicsnemo.cfd.evaluation``."""

from physicsnemo.cfd.bench.metric_registry import (  # noqa: F401
    get_metric,
    list_metrics,
    register_metric,
)

__all__ = ["register_metric", "get_metric", "list_metrics"]
