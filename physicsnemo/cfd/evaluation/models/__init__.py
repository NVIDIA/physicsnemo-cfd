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

"""Registered CFD model wrappers and :class:`CFDModel` registry."""

from physicsnemo.cfd.evaluation.models.model_registry import (
    CFDModel,
    OutputLocation,
    get_inference_domain_for_model,
    get_model_wrapper,
    get_output_location_for_model,
    list_models,
    register_model,
)

# Ensure wrappers register themselves on import
import physicsnemo.cfd.evaluation.models.wrappers  # noqa: F401

__all__ = [
    "CFDModel",
    "OutputLocation",
    "register_model",
    "get_model_wrapper",
    "get_output_location_for_model",
    "get_inference_domain_for_model",
    "list_models",
]
