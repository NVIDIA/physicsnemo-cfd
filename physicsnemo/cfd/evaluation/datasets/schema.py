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


@dataclass
class CanonicalCase:
    """Canonical representation of a single CFD case.

    ``mesh_path`` is the primary mesh: surface ``.vtp`` when ``inference_domain`` is
    ``surface``, or volume ``.vtu`` when ``inference_domain`` is ``volume``.
    """

    case_id: str
    mesh_path: str
    mesh_type: str  # "point" | "cell" — interpretation of field locations
    ground_truth: dict[str, Any] | None = None  # surface: pressure, shear_stress; volume: pressure_volume, …
    metadata: dict[str, Any] = field(default_factory=dict)
    inference_domain: InferenceDomain = "surface"

    def __post_init__(self) -> None:
        if self.mesh_type not in ("point", "cell"):
            raise ValueError("mesh_type must be 'point' or 'cell'")
        if self.inference_domain not in ("surface", "volume"):
            raise ValueError("inference_domain must be 'surface' or 'volume'")

    def get_ground_truth(self, key: str, default: Any = None) -> Any:
        """Return ground truth field by key, or default if missing."""
        if self.ground_truth is None:
            return default
        return self.ground_truth.get(key, default)


def predictions_dict(pressure: np.ndarray, shear_stress: np.ndarray) -> dict[str, np.ndarray]:
    """Build canonical predictions dict from pressure and WSS arrays (surface models)."""
    return {
        "pressure": np.asarray(pressure, dtype=np.float32),
        "shear_stress": np.asarray(shear_stress, dtype=np.float32),
    }


def build_predictions_dict(
    *,
    pressure: np.ndarray | None = None,
    pressure_volume: np.ndarray | None = None,
    shear_stress: np.ndarray | None = None,
    velocity: np.ndarray | None = None,
    turbulent_viscosity: np.ndarray | None = None,
    **extra: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build predictions dict from optional canonical fields (omit None / missing).

    Use ``pressure`` for surface models; ``pressure_volume`` for volume-domain pressure
    (distinct canonical key so surface and volume fields are not conflated).
    """
    out: dict[str, np.ndarray] = {}
    if pressure is not None:
        out["pressure"] = np.asarray(pressure, dtype=np.float32)
    if pressure_volume is not None:
        out["pressure_volume"] = np.asarray(pressure_volume, dtype=np.float32)
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
