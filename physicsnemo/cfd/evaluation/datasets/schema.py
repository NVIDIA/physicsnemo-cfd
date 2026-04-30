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

"""Canonical CFD case schema for dataset adapters and model wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

# Surface vs volume inference (which manifold the case uses). Combined surface+volume is deferred.
InferenceDomain = Literal["surface", "volume"]

_VALID_INFERENCE_DOMAINS = frozenset({"surface", "volume"})


def normalize_inference_domain_str(value: str, *, parameter: str = "inference_domain") -> InferenceDomain:
    """Return ``surface`` or ``volume`` after strip/lowercase; raise on typos."""
    normalized = value.strip().lower()
    if normalized not in _VALID_INFERENCE_DOMAINS:
        raise ValueError(
            f"{parameter} must be 'surface' or 'volume'; got {value!r}"
        )
    return normalized  # type: ignore[return-value]


def coerce_inference_domain_or_default(
    raw: Any,
    *,
    default: InferenceDomain,
    parameter: str,
) -> InferenceDomain:
    """Treat ``None`` as *default*; otherwise validate strings (reject ``None`` sentinel typos separately)."""
    if raw is None:
        return default
    if isinstance(raw, str):
        return normalize_inference_domain_str(raw, parameter=parameter)
    raise ValueError(f"{parameter} must be 'surface', 'volume', or null (got {raw!r})")


@dataclass
class CanonicalCase:
    """Canonical representation of a single CFD case.

    ``mesh_path`` is the primary mesh: surface ``.vtp`` when ``inference_domain`` is
    ``surface``, or volume ``.vtu`` when ``inference_domain`` is ``volume``.
    Decode-time model outputs for metrics should follow :func:`build_predictions_dict` keys/shape.
    """

    case_id: str
    mesh_path: str
    mesh_type: str  # "point" | "cell" — field dof; "unknown" when GT extractor did not set location
    ground_truth: dict[str, Any] | None = None  # surface: pressure, shear_stress; volume: pressure, velocity, …
    metadata: dict[str, Any] = field(default_factory=dict)
    inference_domain: InferenceDomain = "surface"

    def __post_init__(self) -> None:
        if self.mesh_type not in ("point", "cell", "unknown"):
            raise ValueError("mesh_type must be 'point', 'cell', or 'unknown'")
        if self.inference_domain not in ("surface", "volume"):
            raise ValueError("inference_domain must be 'surface' or 'volume'")

    def get_ground_truth(self, key: str, default: Any = None) -> Any:
        """Return ground truth field by key, or default if missing."""
        if self.ground_truth is None:
            return default
        return self.ground_truth.get(key, default)


def build_predictions_dict(
    *,
    pressure: np.ndarray | None = None,
    shear_stress: np.ndarray | None = None,
    velocity: np.ndarray | None = None,
    turbulent_viscosity: np.ndarray | None = None,
    **extra: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build canonical ``decode_outputs`` payloads: numpy arrays as ``float32`` under canonical keys.

    **Use this helper everywhere** in wrapper :meth:`~physicsnemo.cfd.evaluation.models.model_registry.CFDModel.decode_outputs`
    (prefer keyword arguments over ad-hoc ``{...}`` dicts) so dtypes and keys stay consistent.

    Surface wrappers typically pass ``pressure`` and ``shear_stress``; volume wrappers add
    ``velocity`` and ``turbulent_viscosity``. Omit any argument or pass ``None`` to skip keys.
    Extra keyword arrays are normalized the same way (e.g. custom fields).
    """
    out: dict[str, np.ndarray] = {}
    if pressure is not None:
        out["pressure"] = np.asarray(pressure, dtype=np.float32)
    if shear_stress is not None:
        out["shear_stress"] = np.asarray(shear_stress, dtype=np.float32)
    if velocity is not None:
        out["velocity"] = np.asarray(velocity, dtype=np.float32)
    if turbulent_viscosity is not None:
        out["turbulent_viscosity"] = np.asarray(turbulent_viscosity, dtype=np.float32)
    for k, v in extra.items():
        if v is not None:
            out[k] = np.asarray(v, dtype=np.float32)
    return out
