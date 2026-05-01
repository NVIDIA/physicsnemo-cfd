# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for turning composed Hydra configs into :class:`Config` inputs."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def hydra_config_to_benchmark_dict(cfg: DictConfig) -> tuple[dict[str, Any], str | list[str] | None]:
    """
    Convert a composed Hydra ``DictConfig`` into a plain dict for ``Config.from_dict``
    and normalized ``case_id`` (same rules as ``workflows/benchmarking/main.py``).
    """
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("Hydra config root must map to a dict")
    case_id = raw.pop("case_id", None)
    raw.pop("defaults", None)
    raw.pop("hydra", None)
    if case_id is None or case_id == "null":
        case_id = None
    elif isinstance(case_id, (list, tuple)):
        case_id = [str(x) for x in case_id]
    else:
        case_id = str(case_id)
    return raw, case_id
