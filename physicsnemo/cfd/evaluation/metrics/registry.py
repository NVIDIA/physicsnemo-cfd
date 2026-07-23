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

"""Re-export bench metric registry for ``physicsnemo.cfd.evaluation``."""

from physicsnemo.cfd.postprocessing_tools.metric_registry import (  # noqa: F401
    ReducerMetric,
    SampleMetric,
    get_metric,
    is_reducer_metric,
    is_sample_metric,
    list_metrics,
    register_metric,
    unregister_metric,
)

__all__ = [
    "register_metric",
    "unregister_metric",
    "get_metric",
    "list_metrics",
    "ReducerMetric",
    "is_reducer_metric",
    "SampleMetric",
    "is_sample_metric",
]
