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

"""CFDModel base class and registry for model wrappers."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, Type

from physicsnemo.cfd.evaluation.datasets.schema import CanonicalCase, InferenceDomain

# Type aliases for model-specific inputs/outputs (opaque to engine)
ModelInput = Any
RawOutput = Any
Predictions = dict[str, Any]

# Where the model's predictions are defined (mesh points vs cell centers)
OutputLocation = Literal["point", "cell"]

_REGISTRY: dict[str, Type["CFDModel"]] = {}


class CFDModel(ABC):
    """Abstract interface for CFD model wrappers.

    ``INFERENCE_DOMAIN`` is ``surface`` or ``volume``. ``OUTPUT_LOCATION`` is where
    pointwise/cellwise predictions live (``point`` vs ``cell``) on that mesh.
    """

    OUTPUT_LOCATION: ClassVar[OutputLocation]
    INFERENCE_DOMAIN: ClassVar[InferenceDomain] = "surface"

    @property
    @abstractmethod
    def output_location(self) -> OutputLocation:
        """Whether predictions are defined on mesh points or cell centers (primary branch)."""
        ...

    @abstractmethod
    def load(
        self,
        checkpoint_path: str,
        stats_path: str,
        device: str,
        **kwargs: Any,
    ) -> "CFDModel":
        """Load weights and stats; return self for chaining."""
        ...

    @abstractmethod
    def prepare_inputs(self, case: CanonicalCase) -> ModelInput:
        """Turn canonical case into model-specific input (tensors, graph, etc.)."""
        ...

    @abstractmethod
    def predict(self, model_input: ModelInput) -> RawOutput:
        """Run forward pass; return raw model output."""
        ...

    @abstractmethod
    def decode_outputs(self, raw_output: RawOutput, case: CanonicalCase) -> Predictions:
        """Denormalize and map to canonical predictions (e.g. pressure, shear_stress)."""
        ...


def register_model(name: str, wrapper_class: Type[CFDModel]) -> None:
    """Register a model wrapper by name."""
    _REGISTRY[name] = wrapper_class


def get_model_wrapper(name: str) -> Type[CFDModel]:
    """Resolve wrapper class by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_models() -> list[str]:
    """Return registered model names."""
    return list(_REGISTRY.keys())


def get_output_location_for_model(name: str) -> OutputLocation:
    """Return primary output location (``point`` vs ``cell``) without loading weights."""
    cls = get_model_wrapper(name)
    loc = getattr(cls, "OUTPUT_LOCATION", None)
    if loc is not None:
        return loc  # type: ignore[return-value]
    raise ValueError(
        f"Model wrapper {cls.__name__!r} ({name!r}) has no OUTPUT_LOCATION; "
        "cannot align ground truth to model."
    )


def get_inference_domain_for_model(name: str) -> InferenceDomain:
    """Return ``INFERENCE_DOMAIN`` for the registered wrapper without instantiating."""
    cls = get_model_wrapper(name)
    dom = getattr(cls, "INFERENCE_DOMAIN", "surface")
    if dom in ("surface", "volume"):
        return dom  # type: ignore[return-value]
    return "surface"
