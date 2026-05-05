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

"""Load DoMINO scaling factors from training pickles (compatible with domino ``utils.ScalingFactors``)."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import DictConfig


@dataclass
class ScalingFactors:
    """Mirrors ``examples/cfd/external_aerodynamics/domino/src/utils.ScalingFactors`` for unpickling."""

    mean: Dict[str, np.ndarray]
    std: Dict[str, np.ndarray]
    min_val: Dict[str, np.ndarray]
    max_val: Dict[str, np.ndarray]
    field_keys: list[str]

    @classmethod
    def load(cls, filepath: str | Path) -> "ScalingFactors":
        """Load a pickled :class:`ScalingFactors` instance from ``filepath``."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


class _ScalingUnpickler(pickle.Unpickler):
    """Map pickles saved as ``utils.ScalingFactors`` to our local class."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "utils" and name == "ScalingFactors":
            return ScalingFactors
        return super().find_class(module, name)


# Only import/layout compatibility: do not catch ``pickle.UnpicklingError`` or ``Exception``,
# or a restricted-load failure would fall through to unrestricted ``pickle.load`` (unsafe).
_COMPAT_UNPICKLE_ERRORS: tuple[type[BaseException], ...] = (
    AttributeError,
    ImportError,
)


def load_scaling_factors_tensors(
    cfg: DictConfig,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Return ``(vol_factors, surf_factors)`` as in domino ``test.py`` / ``load_scaling_factors``."""
    pickle_path = os.path.normpath(os.path.expanduser(str(cfg.data.scaling_factors)))
    _p = Path(pickle_path)
    # Stale YAML sometimes points at ``global_stats.json``; DoMINO needs ``scaling_factors.pkl``.
    if _p.is_file() and _p.suffix.lower() == ".json":
        _alt = (_p.parent / "scaling_factors.pkl").resolve()
        if _alt.is_file():
            pickle_path = str(_alt)
        else:
            raise ValueError(
                "DoMINO ``data.scaling_factors`` must be ``scaling_factors.pkl``, not JSON. "
                f"No scaling_factors.pkl beside {_p}."
            )
    with open(pickle_path, "rb") as f:
        try:
            scaling_factors = _ScalingUnpickler(f).load()
        except _COMPAT_UNPICKLE_ERRORS:
            f.seek(0)
            scaling_factors = pickle.load(f)

    if cfg.model.normalization == "min_max_scaling":
        vol_factors = np.asarray(
            [
                scaling_factors.max_val["volume_fields"],
                scaling_factors.min_val["volume_fields"],
            ]
        )
        surf_factors = np.asarray(
            [
                scaling_factors.max_val["surface_fields"],
                scaling_factors.min_val["surface_fields"],
            ]
        )
    elif cfg.model.normalization == "mean_std_scaling":
        vol_factors = np.asarray(
            [
                scaling_factors.mean["volume_fields"],
                scaling_factors.std["volume_fields"],
            ]
        )
        surf_factors = np.asarray(
            [
                scaling_factors.mean["surface_fields"],
                scaling_factors.std["surface_fields"],
            ]
        )
    else:
        raise ValueError(f"Invalid normalization mode: {cfg.model.normalization}")

    vol_t = torch.from_numpy(vol_factors).to(device=device, dtype=torch.float32)
    surf_t = torch.from_numpy(surf_factors).to(device=device, dtype=torch.float32)

    if cfg.model.model_type == "surface":
        vol_t = None
    return vol_t, surf_t
