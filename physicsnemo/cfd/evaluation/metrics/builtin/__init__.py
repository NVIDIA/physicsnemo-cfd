# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register built-in evaluation metrics backed by physicsnemo.cfd.postprocessing_tools."""

from physicsnemo.cfd.evaluation.metrics.builtin.forces import register_force_metrics
from physicsnemo.cfd.evaluation.metrics.builtin.l2 import register_l2_metrics
from physicsnemo.cfd.evaluation.metrics.builtin.physics import register_physics_metrics


def register_all_builtin_metrics() -> None:
    """Idempotent: register all default metric names."""
    register_l2_metrics()
    register_force_metrics()
    register_physics_metrics()


register_all_builtin_metrics()
