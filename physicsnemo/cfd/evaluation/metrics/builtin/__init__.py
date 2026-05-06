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
