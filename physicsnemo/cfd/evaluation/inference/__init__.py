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

"""Inference CLI and progress helpers.

Model wrappers and registration live in :mod:`physicsnemo.cfd.evaluation.models`.
This package re-exports the registry API for backward compatibility.
"""

from physicsnemo.cfd.evaluation.inference.progress import log_inference
from physicsnemo.cfd.evaluation.models import (
    CFDModel,
    OutputLocation,
    get_inference_domain_for_model,
    get_model_wrapper,
    get_output_location_for_model,
    list_models,
    register_model,
)

import physicsnemo.cfd.evaluation.models.wrappers  # noqa: F401

__all__ = [
    "CFDModel",
    "OutputLocation",
    "log_inference",
    "register_model",
    "get_model_wrapper",
    "get_output_location_for_model",
    "get_inference_domain_for_model",
    "list_models",
]
