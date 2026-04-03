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

"""DoMINO surface inference: wrapper + colocated helpers."""

from physicsnemo.cfd.evaluation.inference.wrappers.domino.inference import (
    build_domin_surface_datadict,
    domino_count_output_features,
    domino_surface_test_step,
)
from physicsnemo.cfd.evaluation.inference.wrappers.domino.scaling import load_scaling_factors_tensors
from physicsnemo.cfd.evaluation.inference.wrappers.domino.wrapper import DominoWrapper

__all__ = [
    "DominoWrapper",
    "build_domin_surface_datadict",
    "domino_count_output_features",
    "domino_surface_test_step",
    "load_scaling_factors_tensors",
]
