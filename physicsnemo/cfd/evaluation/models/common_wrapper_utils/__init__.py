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

"""Shared helpers for inference model wrappers (VTK/STL, datapipe tensors, etc.)."""

from physicsnemo.cfd.evaluation.models.common_wrapper_utils.vtk_datapipe_io import (
    build_surface_data_dict,
    build_volume_data_dict,
    read_stl_geometry,
    read_surface_from_vtp,
    read_volume_from_vtu,
    run_id_from_case_id,
)

__all__ = [
    "build_surface_data_dict",
    "build_volume_data_dict",
    "read_stl_geometry",
    "read_surface_from_vtp",
    "read_volume_from_vtu",
    "run_id_from_case_id",
]
