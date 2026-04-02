"""Model inference and wrappers."""

from physicsnemo.cfd.evaluation.inference.model_registry import (
    CFDModel,
    OutputLocation,
    get_inference_domain_for_model,
    get_model_wrapper,
    get_output_location_for_model,
    list_models,
    register_model,
)
# Ensure wrappers register themselves
import physicsnemo.cfd.evaluation.inference.wrappers  # noqa: F401

__all__ = [
    "CFDModel",
    "OutputLocation",
    "register_model",
    "get_model_wrapper",
    "get_output_location_for_model",
    "get_inference_domain_for_model",
    "list_models",
]
