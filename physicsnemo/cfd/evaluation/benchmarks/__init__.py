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

"""Benchmark engine and reporting.

Submodules such as ``metrics_cache`` can be imported without loading ``engine`` (PEP 562).
"""

from __future__ import annotations

from typing import Any

__all__ = ["BenchmarkPolicyError", "run_benchmark", "write_report"]


def __getattr__(name: str) -> Any:
    if name == "BenchmarkPolicyError":
        from physicsnemo.cfd.evaluation.benchmarks.engine import BenchmarkPolicyError

        return BenchmarkPolicyError
    if name == "run_benchmark":
        from physicsnemo.cfd.evaluation.benchmarks.engine import run_benchmark

        return run_benchmark
    if name == "write_report":
        from physicsnemo.cfd.evaluation.benchmarks.report import write_report

        return write_report
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
