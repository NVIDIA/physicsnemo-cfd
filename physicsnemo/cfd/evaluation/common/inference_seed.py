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

"""Deterministic RNG seeding for per-case inference (subsampling / ``randperm`` in wrappers).

``run_benchmark`` derives a stable seed per ``(run.seed, case_id)`` via SHA256 and sets
numpy, Python :mod:`random`, and torch (including CUDA) before ``prepare_inputs`` → ``predict``.
This makes xmgn/FiG subsampling and GeoTransolver/Transolver ``randperm`` order reproducible,
independent of which other cases ran earlier in the process.
"""

from __future__ import annotations

import hashlib
import random


def inference_seed(run_seed: int, case_id: str) -> int:
    """Return a nonnegative 63-bit stable integer from ``run_seed`` and ``case_id``.

    Uses SHA256 over ``run_seed`` and UTF-8 ``case_id`` (no salted :func:`hash` — stable
    across processes and Python versions).
    """
    payload = str(int(run_seed)).encode("ascii") + b"\x00" + case_id.encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFFFFFFFFFF


def seed_inference_rng(run_seed: int, case_id: str) -> int:
    """Set numpy / ``random`` / torch global RNG state for one benchmark case."""
    seed = inference_seed(run_seed, case_id)
    numpy_seed = seed & 0xFFFFFFFF

    import numpy as np
    import torch

    np.random.seed(numpy_seed)
    random.seed(seed)
    # torch.manual_seed truncates internally; avoids numpy/torch bitwise overlap
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed
