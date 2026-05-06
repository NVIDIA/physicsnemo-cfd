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

"""Device-aware bf16 autocast for inference wrappers (CUDA only; explicit no-op on CPU)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Union

import torch


@contextmanager
def cuda_bf16_autocast(device: Union[str, torch.device]) -> Iterator[None]:
    """bf16 autocast on CUDA; disabled autocast context on CPU (never forces device_type='cuda')."""
    dev = device if isinstance(device, torch.device) else torch.device(str(device))
    device_type = dev.type if dev.type in ("cuda", "cpu") else "cpu"
    use_cuda = dev.type == "cuda"
    with torch.autocast(
        device_type=device_type,
        enabled=use_cuda,
        dtype=torch.bfloat16,
    ):
        yield
