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

"""Stub adapter for Ahmed Body dataset (placeholder until implementation)."""

from pathlib import Path

from physicsnemo.cfd.evaluation.datasets.adapter_registry import DatasetAdapter
from physicsnemo.cfd.evaluation.datasets.schema import CanonicalCase


class AhmedAdapter(DatasetAdapter):
    """Placeholder for Ahmed Body dataset. Implement list_cases and load_case."""

    def __init__(self, root: str, **kwargs: object) -> None:
        self.root = Path(root)
        if self.root.exists():
            pass  # optional: validate layout

    def list_cases(self, split: str | None = None) -> list[str]:
        """Return case IDs when dataset is implemented."""
        if not self.root.exists():
            return []
        # Stub: no cases until real layout is defined
        return []

    def load_case(self, case_id: str) -> CanonicalCase:
        """Load one case. Raises until dataset layout is implemented."""
        raise NotImplementedError(
            "Ahmed Body adapter not yet implemented. "
            "Add mesh layout and GT loading in datasets/adapters/ahmed.py"
        )
