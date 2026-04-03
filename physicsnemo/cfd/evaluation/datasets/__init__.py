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

"""Dataset adapters and canonical case schema."""

from physicsnemo.cfd.evaluation.datasets.schema import CanonicalCase, predictions_dict
from physicsnemo.cfd.evaluation.datasets.adapter_registry import (
    get_adapter,
    list_adapters,
    register_adapter,
    DatasetAdapter,
)
import physicsnemo.cfd.evaluation.datasets.adapters  # noqa: F401 - register drivaerml, etc.

__all__ = [
    "CanonicalCase",
    "predictions_dict",
    "DatasetAdapter",
    "register_adapter",
    "get_adapter",
    "list_adapters",
]
