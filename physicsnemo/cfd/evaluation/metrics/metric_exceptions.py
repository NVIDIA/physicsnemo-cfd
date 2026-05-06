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

"""Exception groups for recoverable mesh-metric failures (log + NaN / scalar fallback)."""

from __future__ import annotations

_RECOVERABLE_BASE: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    IndexError,
    ArithmeticError,
    OSError,
    MemoryError,
    RuntimeError,
)


def _pyvista_metric_recovery_types() -> tuple[type[BaseException], ...]:
    """PyVista / VTK errors during mesh ops (aligned with ``benchmarks.engine`` metric recovery)."""
    try:
        import pyvista as pv  # noqa: PLC0415
    except ImportError:
        return ()
    names = (
        "AmbiguousDataError",
        "InvalidMeshError",
        "MissingDataError",
        "VTKExecutionError",
        "VTKVersionError",
        "PointSetCellOperationError",
        "PyVistaAttributeError",
        "PyVistaPipelineError",
        "NotAllTrianglesError",
    )
    tt: list[type[BaseException]] = []
    for name in names:
        obj = getattr(pv, name, None)
        if isinstance(obj, type) and issubclass(obj, BaseException):
            tt.append(obj)
    return tuple(tt)


RECOVERABLE_MESH_METRIC_ERRORS: tuple[type[BaseException], ...] = (
    _RECOVERABLE_BASE + _pyvista_metric_recovery_types()
)
