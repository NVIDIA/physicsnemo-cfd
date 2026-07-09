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
from typing import Any, ClassVar, Literal, Optional, Type

from physicsnemo.cfd.evaluation.datasets.schema import (
    CanonicalCase,
    FieldDistribution,
    InferenceDomain,
    normalize_inference_domain_str,
)

# Type aliases for model-specific inputs/outputs (opaque to engine)
ModelInput = Any
RawOutput = Any
Predictions = dict[str, Any]

# Where the model's predictions are defined (mesh points vs cell centers)
OutputLocation = Literal["point", "cell"]

_REGISTRY: dict[str, Type["CFDModel"]] = {}


class CFDModel(ABC):
    """Abstract interface for CFD model wrappers.

    ``INFERENCE_DOMAIN`` is ``surface``, ``volume``, or ``None``. Use ``None`` for
    wrappers that support both manifolds; routing then uses ``model.(kwargs.)inference_domain``
    and/or :meth:`inference_domain_from_kwargs`. ``OUTPUT_LOCATION`` is where
    pointwise/cellwise predictions live (``point`` vs ``cell``) on that mesh.

    Set ``REQUIRES_REMOTE_ASSETS = False`` on stubs (e.g. baselines) that do not need
    ``checkpoint`` / ``stats_path`` or a Hugging Face ``package``.
    """

    OUTPUT_LOCATION: ClassVar[OutputLocation]
    INFERENCE_DOMAIN: ClassVar[InferenceDomain | None] = "surface"
    REQUIRES_REMOTE_ASSETS: ClassVar[bool] = True

    #: Whether this wrapper produces a predictive *distribution* (uncertainty), not just a
    #: point estimate. Deterministic wrappers leave this ``False``; UQ metrics then report
    #: ``NaN`` for them (consistent with the engine's recoverable-metric behavior).
    SUPPORTS_UQ: ClassVar[bool] = False
    #: How the predictive distribution is produced (see the UQ design doc §4.2):
    #: ``"analytic"`` — one forward pass emits the distribution/params (GP, mean-variance,
    #: evidential); the wrapper overrides :meth:`decode_distribution`.
    #: ``"sampling"`` — the distribution is built from statistics over ``N`` stochastic
    #: passes / ensemble members; the engine drives the passes and aggregates.
    #: ``"none"`` — deterministic.
    UQ_METHOD: ClassVar[Literal["none", "analytic", "sampling"]] = "none"

    @classmethod
    def inference_domain_from_kwargs(
        cls, kwargs: dict[str, Any]
    ) -> InferenceDomain | None:
        """Deduce ``surface``/``volume`` before :meth:`load` when ``model.inference_domain`` is omitted.

        Return ``None`` to fall back to a **fixed** :attr:`INFERENCE_DOMAIN` on the class.
        Dual-mode wrappers set :attr:`INFERENCE_DOMAIN` to ``None`` and should implement
        this method (or require ``inference_domain`` in merged kwargs) so routing does
        not assume ``surface`` from a misleading class default.
        Called with :meth:`~physicsnemo.cfd.evaluation.config.ModelConfig.merged_kwargs_for_load`.
        """
        return None

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
    def decode_outputs(
        self,
        raw_output: RawOutput,
        case: CanonicalCase,
        model_input: Optional[ModelInput] = None,
    ) -> Predictions:
        """Denormalize and map to canonical predictions (e.g. pressure, shear_stress).

        Pass the same ``model_input`` returned by :meth:`prepare_inputs` when decode must
        align with inference geometry (e.g. interpolation / subsampling in xmgn/fignet).
        """
        ...

    def decode_distribution(
        self,
        raw_output: RawOutput,
        case: CanonicalCase,
        model_input: Optional[ModelInput] = None,
    ) -> dict[str, "FieldDistribution"]:
        """Map raw output to a per-field predictive distribution (physical units).

        **Analytic** UQ wrappers (``UQ_METHOD="analytic"``, e.g. a GP head) override this to
        return :class:`~physicsnemo.cfd.evaluation.datasets.schema.FieldDistribution` with a
        real ``std`` / ``epistemic_std``. The default wraps :meth:`decode_outputs` as
        **degenerate** distributions (``std=None``) so callers can uniformly request a
        distribution from any wrapper.
        """
        preds = self.decode_outputs(raw_output, case, model_input)
        return {k: FieldDistribution(mean=v) for k, v in preds.items()}

    def predict_ensemble(
        self, model_input: ModelInput, n: int
    ) -> Optional[list[RawOutput]]:
        """Optional fast-path for ``UQ_METHOD="sampling"`` wrappers.

        Return ``n`` raw outputs (stochastic passes or ensemble members) in one call, letting
        the engine skip its per-pass Python loop. **Only implement this when all ``n`` raw
        outputs fit simultaneously in the compute device's memory (typically GPU), since the
        returned list holds them all at once** — for full-mesh CFD fields that is usually ``n×``
        the single-pass footprint and will OOM for large ``n``.

        Return ``None`` (the default, used by all shipped sampling wrappers) to have the engine
        instead call :meth:`predict` ``n`` times and stream the passes through Welford
        aggregation (:func:`~physicsnemo.cfd.evaluation.benchmarks.uq_inference.run_sampling_inference`):
        only **one** raw output is resident on the GPU at a time, and the running mean/variance
        accumulators are a handful of host (CPU) arrays of one field's size — i.e. memory is
        O(1) in ``n``. Prefer this default unless you have measured that the batched path fits.
        """
        return None


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
    """Return validated ``surface`` / ``volume`` from the wrapper's ``INFERENCE_DOMAIN``.

    Missing ``INFERENCE_DOMAIN`` on the class uses :func:`getattr` with default
    ``"surface"`` (same base default as :class:`CFDModel`). Explicit ``None`` marks a dual-mode
    wrapper (**no** static domain); callers must not use this fallback — use merged kwargs /
    :meth:`~CFDModel.inference_domain_from_kwargs` in :func:`benchmarks.engine._effective_inference_domain`
    instead. Passing :func:`get_inference_domain_for_model` for such a wrapper raises :exc:`ValueError`.

    Non-``None`` values pass through
    :func:`~physicsnemo.cfd.evaluation.datasets.schema.normalize_inference_domain_str`;
    typos raise :exc:`ValueError`.
    """
    cls = get_model_wrapper(name)
    dom = getattr(cls, "INFERENCE_DOMAIN", "surface")
    if dom is None:
        raise ValueError(
            f"{cls.__name__}.INFERENCE_DOMAIN is None (dual-mode wrapper). "
            "Do not use get_inference_domain_for_model for routing; set "
            "``model.inference_domain`` / ``model.kwargs.inference_domain`` or use "
            f"``{cls.__name__}.inference_domain_from_kwargs`` with merged kwargs."
        )
    return normalize_inference_domain_str(
        dom if isinstance(dom, str) else str(dom),
        parameter=f"{cls.__name__}.INFERENCE_DOMAIN",
    )
