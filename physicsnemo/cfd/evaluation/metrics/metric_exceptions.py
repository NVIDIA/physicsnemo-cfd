# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
