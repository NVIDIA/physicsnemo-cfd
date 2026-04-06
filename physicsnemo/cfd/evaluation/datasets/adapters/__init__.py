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

"""Dataset adapter implementations."""

from physicsnemo.cfd.evaluation.datasets.adapter_registry import register_adapter
from physicsnemo.cfd.evaluation.datasets.adapters.drivaerml import DrivAerMLAdapter
from physicsnemo.cfd.evaluation.datasets.adapters.ahmed import AhmedAdapter

register_adapter("drivaerml", DrivAerMLAdapter)
register_adapter("ahmed", AhmedAdapter)

__all__ = ["DrivAerMLAdapter", "AhmedAdapter"]
