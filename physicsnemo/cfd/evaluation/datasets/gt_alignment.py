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

"""Resolve ground-truth mesh location (cell vs point) from dataset kwargs + model name."""

from __future__ import annotations

from typing import Any

from physicsnemo.cfd.evaluation.config import _parse_bool
from physicsnemo.cfd.evaluation.models.model_registry import (
    get_output_location_for_model,
)

# Keys used only for resolution; stripped before passing kwargs to dataset adapters.
_ALIGN_KEYS = frozenset({"align_ground_truth_to_model"})


def resolve_dataset_kwargs_for_model(
    dataset_kwargs: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """Return a copy of ``dataset_kwargs`` with effective ``gt_data_type`` when requested.

    Raises ``ValueError`` if ``split`` appears in kwargs (benchmarks iterate all cases under
    ``dataset.root``, optionally narrowed via ``dataset.case_ids``).

    **Precedence**

    - Explicit ``gt_data_type`` / ``gt_prefer`` of ``cell`` or ``point`` always wins.
    - Otherwise, if ``gt_data_type`` is ``from_model`` / ``model`` (case-insensitive), or
      ``align_ground_truth_to_model: true``, set ``gt_data_type`` to the registered model's
      :attr:`~physicsnemo.cfd.evaluation.models.model_registry.CFDModel.OUTPUT_LOCATION`.
    - Otherwise leave ``gt_data_type`` unchanged (e.g. ``auto``).

    Alignment keys are removed from the returned dict so adapters only see concrete
    ``auto`` | ``cell`` | ``point``.
    """
    if dataset_kwargs.get("split") is not None:
        raise ValueError(
            "dataset.kwargs.split is not supported — evaluation uses all cases under dataset.root "
            "(or dataset.case_ids / case_id overrides). Remove 'split' from dataset kwargs."
        )
    kw = dict(dataset_kwargs)
    gt_raw = kw.get("gt_data_type", kw.get("gt_prefer", "auto"))
    align = _parse_bool(kw.get("align_ground_truth_to_model"), default=False)

    explicit = {"cell", "point"}
    if isinstance(gt_raw, str) and gt_raw.lower() in explicit:
        kw["gt_data_type"] = gt_raw.lower()
        _pop_align_keys(kw)
        return kw

    from_model_aliases = {"from_model", "model"}
    use_model = align or (
        isinstance(gt_raw, str) and gt_raw.lower() in from_model_aliases
    )
    if use_model:
        loc = get_output_location_for_model(model_name)
        kw["gt_data_type"] = loc
        _pop_align_keys(kw)
        return kw

    _pop_align_keys(kw)
    return kw


def _pop_align_keys(kw: dict[str, Any]) -> None:
    for k in _ALIGN_KEYS:
        kw.pop(k, None)
