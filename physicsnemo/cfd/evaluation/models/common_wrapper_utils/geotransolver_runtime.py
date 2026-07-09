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

"""Shared GeoTransolver + ``TransolverDataPipe`` runtime helpers (composition, not inheritance).

The GeoTransolver-family wrappers are **independent** :class:`CFDModel` subclasses (no shared base
beyond ``CFDModel``): the DrivAerML baseline, the ``transformer_models`` deterministic wrapper, the
GP-head wrapper, and the MC weight-perturbation proxy. They share their plumbing through the free
functions here rather than class inheritance:

* :func:`build_geotransolver_backbone` — construct the backbone and load its checkpoint.
* :func:`parse_runtime_kwargs` — pull the common inference kwargs into a :class:`GeoTransolverRuntimeConfig`.
* :func:`build_transolver_batch` — read a case, (re)build the datapipe, and produce the model batch.
* :func:`geotransolver_forward` / :func:`unscale_targets` — the blocked forward pass + target unscaling.
* :func:`decode_surface_predictions` / :func:`decode_volume_predictions` — physical-unit decode.

The ``REDIMENSIONALIZE_OUTPUTS`` decision (whether to re-dimensionalize by ``ρu²`` / ``u`` / ``u·L``)
is owned by each wrapper and passed explicitly into the decode helpers.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from physicsnemo.cfd.evaluation.common.checkpoint_compat import (
    parse_checkpoint_epoch,
    trusted_torch_load_context,
)
from physicsnemo.cfd.evaluation.config import _parse_bool
from physicsnemo.cfd.evaluation.datasets.schema import (
    CanonicalCase,
    build_predictions_dict,
)
from physicsnemo.cfd.evaluation.inference.progress import log_inference
from physicsnemo.cfd.evaluation.models.common_wrapper_utils.vtk_datapipe_io import (
    build_surface_data_dict,
    build_volume_data_dict,
    run_id_from_case_id,
)
from physicsnemo.cfd.evaluation.models.inference_autocast import cuda_bf16_autocast
from physicsnemo.cfd.evaluation.models.model_registry import Predictions, RawOutput

# Optional physicsnemo imports (required for real inference).
try:
    from physicsnemo.datapipes.cae.transolver_datapipe import TransolverDataPipe
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.experimental.models.geotransolver import GeoTransolver
    from physicsnemo.utils import load_checkpoint

    _PHYSICSNEMO_AVAILABLE = True
except ImportError:
    _PHYSICSNEMO_AVAILABLE = False


# Default GeoTransolver surface config (from geotransolver_surface + model/geotransolver.yaml).
DEFAULT_GEOTRANSOLVER_KW = dict(
    functional_dim=6,
    global_dim=2,
    geometry_dim=3,
    out_dim=4,
    n_layers=20,
    n_hidden=256,
    dropout=0.0,
    n_head=8,
    act="gelu",
    mlp_ratio=2,
    slice_num=128,
    use_te=False,
    plus=False,
    include_local_features=True,
    radii=[0.01, 0.05, 0.25, 1.0, 2.5, 5.0],
    neighbors_in_radius=[4, 8, 16, 64, 128, 256],
    n_hidden_local=32,
)

# Volume training defaults (geotransolver_volume.yaml).
DEFAULT_GEOTRANSOLVER_VOLUME_KW = {
    **DEFAULT_GEOTRANSOLVER_KW,
    "functional_dim": 7,
    "out_dim": 5,
}

#: ``model.kwargs`` keys forwarded to ``TransolverDataPipe`` (overrides of the static defaults).
DATAPIPE_KEYS = frozenset(
    {
        "include_normals",
        "include_sdf",
        "translational_invariance",
        "scale_invariance",
        "reference_scale",
        "broadcast_global_features",
        "include_geometry",
        "return_mesh_features",
    }
)


def geotransolver_available() -> bool:
    """True when physicsnemo (GeoTransolver, TransolverDataPipe, load_checkpoint) is importable."""
    return _PHYSICSNEMO_AVAILABLE


def global_fx_to_bnc(fx: torch.Tensor) -> torch.Tensor:
    """GeoTransolver requires ``global_embedding`` shape (B, N_g, C_g).

    With ``broadcast_global_features=False``, ``TransolverDataPipe`` may stack ``fx`` as
    (1, 1, C) before ``__call__`` adds another batch dimension, yielding (1, 1, 1, C).
    Squeeze singleton middle dims until 3D.
    """
    out = fx
    while out.ndim > 3:
        for d in range(1, out.ndim - 1):
            if out.shape[d] == 1:
                out = out.squeeze(d)
                break
        else:
            raise ValueError(
                "Cannot reshape global ``fx`` to 3D for GeoTransolver; "
                f"shape={tuple(fx.shape)}"
            )
    return out


def _surface_datapipe_static_kw() -> dict[str, Any]:
    return dict(
        include_normals=True,
        include_sdf=False,
        broadcast_global_features=False,
        include_geometry=True,
        translational_invariance=True,
        scale_invariance=True,
        reference_scale=[12.0, 4.5, 3.25],
        return_mesh_features=True,
    )


def _volume_datapipe_static_kw() -> dict[str, Any]:
    """Defaults aligned with ``data/core.yaml`` + ``geotransolver_volume.yaml``."""
    return dict(
        include_normals=True,
        include_sdf=True,
        translational_invariance=True,
        scale_invariance=True,
        reference_scale=[12.0, 4.5, 3.25],
        broadcast_global_features=False,
        include_geometry=True,
        return_mesh_features=False,
    )


@dataclass
class GeoTransolverRuntimeConfig:
    """Common inference knobs parsed once in ``load`` and reused by every runtime helper."""

    device: str = "cuda:0"
    air_density: float = 1.205
    stream_velocity: float = 30.0
    batch_resolution: int = 2048
    cuda_bf16_autocast: bool = False
    geometry_sampling: int = 300_000
    datapipe_user_kw: dict[str, Any] = field(default_factory=dict)


def split_datapipe_kwargs(kw: dict[str, Any]) -> dict[str, Any]:
    """Pop ``TransolverDataPipe`` overrides out of a mutable ``model.kwargs`` dict."""
    return {k: kw.pop(k) for k in list(kw.keys()) if k in DATAPIPE_KEYS}


def parse_runtime_kwargs(kw: dict[str, Any], device: str) -> GeoTransolverRuntimeConfig:
    """Parse the shared runtime kwargs into a :class:`GeoTransolverRuntimeConfig` (mutates ``kw``).

    Pops ``cuda_bf16_autocast``, the datapipe override keys, and the ignored ``resolution`` from
    ``kw`` (benchmark inference always uses the full mesh, ``resolution=None``); the remaining
    entries in ``kw`` are read non-destructively so a caller can inspect them afterwards.
    """
    cfg = GeoTransolverRuntimeConfig(
        device=device,
        air_density=float(kw.get("air_density", 1.205)),
        stream_velocity=float(kw.get("stream_velocity", 30.0)),
        batch_resolution=int(kw.get("batch_resolution", 2048)),
        cuda_bf16_autocast=_parse_bool(
            kw.pop("cuda_bf16_autocast", None), default=False
        ),
        geometry_sampling=int(kw.get("geometry_sampling", 300_000)),
    )
    # Benchmark inference uses all mesh points; ``resolution`` in model kwargs is ignored.
    kw.pop("resolution", None)
    cfg.datapipe_user_kw = split_datapipe_kwargs(kw)
    return cfg


def ensure_distributed_initialized() -> None:
    """Initialize ``DistributedManager`` once (``load_checkpoint`` relies on it)."""
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()


def resolve_checkpoint_file(checkpoint_path: str) -> tuple[Path, int]:
    """Resolve a checkpoint knob to ``(directory, epoch)`` from a **specific file name**.

    The config must point ``checkpoint`` at a concrete checkpoint file whose name encodes the
    epoch (``<Model>.0.<epoch>.mdlus`` or ``checkpoint.0.<epoch>.pt``) rather than a directory.
    This makes the loaded epoch a single source of truth (no silent "latest epoch" fallback and
    no drift between the backbone and a separately-loaded head). ``physicsnemo.load_checkpoint``
    itself takes the parent directory plus the epoch, which are both derived here.
    """
    ckpt = Path(checkpoint_path)
    if not checkpoint_path:
        raise ValueError("checkpoint path is empty; point it at a specific checkpoint file.")
    if ckpt.is_dir():
        raise ValueError(
            f"checkpoint {checkpoint_path!r} is a directory; point it at a specific checkpoint "
            "file whose name encodes the epoch (e.g. ``GeoTransolver.0.30.mdlus`` or "
            "``checkpoint.0.100.pt``) so the loaded epoch is unambiguous."
        )
    epoch = parse_checkpoint_epoch(checkpoint_path)
    if epoch is None:
        raise ValueError(
            f"cannot parse an epoch from checkpoint file name {ckpt.name!r}; expected "
            "``<Model>.0.<epoch>.mdlus`` or ``checkpoint.0.<epoch>.pt``."
        )
    return ckpt.parent, epoch


def build_geotransolver_backbone(
    *, checkpoint_path: str, device: str, inference_mode: str
) -> "GeoTransolver":
    """Construct a ``GeoTransolver`` for ``surface``/``volume`` and load its checkpoint onto ``device``.

    ``checkpoint_path`` must be a specific checkpoint file (see :func:`resolve_checkpoint_file`);
    a directory or an epoch-less name is rejected rather than silently loading the latest epoch.
    """
    model_kw = dict(
        DEFAULT_GEOTRANSOLVER_VOLUME_KW
        if inference_mode == "volume"
        else DEFAULT_GEOTRANSOLVER_KW
    )
    checkpoint_dir, epoch = resolve_checkpoint_file(checkpoint_path)

    ensure_distributed_initialized()
    dev = torch.device(device)
    model = GeoTransolver(**model_kw)
    with trusted_torch_load_context():
        _ = load_checkpoint(
            path=str(checkpoint_dir), models=model, epoch=epoch, device=dev
        )
    model = model.to(dev)
    model.eval()
    return model


def _move_reference_scale_to_device(dp: "TransolverDataPipe", device: str) -> None:
    """``reference_scale`` is often created on CPU; mesh tensors live on ``device``."""
    if dp.config.scale_invariance and dp.config.reference_scale is not None:
        dp.config.reference_scale = dp.config.reference_scale.to(torch.device(device))


def build_surface_datapipe(
    *, geometry_sampling: int, surface_factors: Any, device: str, user_kw: dict[str, Any]
) -> "TransolverDataPipe":
    """Build a surface ``TransolverDataPipe`` (static defaults merged with ``user_kw``)."""
    merged = {**_surface_datapipe_static_kw(), **user_kw}
    merged["geometry_sampling"] = geometry_sampling
    dp = TransolverDataPipe(
        input_path=None,
        model_type="surface",
        resolution=None,
        surface_factors=surface_factors,
        volume_factors=None,
        scaling_type="mean_std_scaling",
        **merged,
    )
    _move_reference_scale_to_device(dp, device)
    return dp


def build_volume_datapipe(
    *, geometry_sampling: int, volume_factors: Any, device: str, user_kw: dict[str, Any]
) -> "TransolverDataPipe":
    """Build a volume ``TransolverDataPipe`` (static defaults merged with ``user_kw``)."""
    merged = {**_volume_datapipe_static_kw(), **user_kw}
    merged["geometry_sampling"] = geometry_sampling
    dp = TransolverDataPipe(
        input_path=None,
        model_type="volume",
        resolution=None,
        surface_factors=None,
        volume_factors=volume_factors,
        scaling_type="mean_std_scaling",
        **merged,
    )
    _move_reference_scale_to_device(dp, device)
    return dp


def _volume_length_scale(data_dict: dict[str, Any]) -> float:
    """STL bounding-box max extent (matches DoMINO ``length_scale``), used to unscale νₜ by ``u·L``."""
    stl = data_dict["stl_coordinates"]
    return float((stl.amax(dim=0) - stl.amin(dim=0)).max().item())


@dataclass
class TransolverBatch:
    """Result of :func:`build_transolver_batch`: the model batch plus the (cached) datapipe."""

    batch: dict[str, Any]
    datapipe: "TransolverDataPipe"
    geometry_effective: int
    volume_length_scale: float | None = None


def build_transolver_batch(
    *,
    case: CanonicalCase,
    inference_mode: str,
    cfg: GeoTransolverRuntimeConfig,
    surface_factors: Any,
    volume_factors: Any,
    datapipe: "TransolverDataPipe | None",
    geometry_effective: int | None,
) -> TransolverBatch:
    """Read a case, (re)build the datapipe when the effective sampling changed, and run it.

    ``datapipe`` / ``geometry_effective`` are the caller's cached state; pass the returned
    :class:`TransolverBatch` fields back next time to reuse the datapipe across cases of equal size.
    """
    log_inference(
        "geotransolver",
        f"Reading case inputs (case {case.case_id}): mesh {case.mesh_path}, "
        f"run dir {Path(case.mesh_path).parent}",
    )
    run_dir = Path(case.mesh_path).parent
    run_idx = run_id_from_case_id(case.case_id)
    device = torch.device(cfg.device)
    length_scale: float | None = None
    # In-memory geometry (surface == geometry for e.g. DrivAerStar) bypasses the STL file glob.
    geometry_mesh = getattr(case, "geometry", None)

    if inference_mode == "volume":
        data_dict = build_volume_data_dict(
            run_dir=run_dir,
            vtu_path=case.mesh_path,
            device=device,
            air_density=cfg.air_density,
            stream_velocity=cfg.stream_velocity,
            run_idx=run_idx,
            reference_mesh=case.reference_geometry,
            geometry_mesh=geometry_mesh,
        )
        length_scale = _volume_length_scale(data_dict)
        n_stl = int(data_dict["stl_coordinates"].shape[0])
        safe_geo = max(1, min(cfg.geometry_sampling, n_stl))
        if datapipe is None or geometry_effective != safe_geo:
            datapipe = build_volume_datapipe(
                geometry_sampling=safe_geo,
                volume_factors=volume_factors,
                device=cfg.device,
                user_kw=cfg.datapipe_user_kw,
            )
            geometry_effective = safe_geo
    else:
        data_dict = build_surface_data_dict(
            run_dir=run_dir,
            vtp_path=case.mesh_path,
            device=device,
            air_density=cfg.air_density,
            stream_velocity=cfg.stream_velocity,
            run_idx=run_idx,
            reference_mesh=case.reference_geometry,
            geometry_mesh=geometry_mesh,
        )
        n_stl = int(data_dict["stl_coordinates"].shape[0])
        n_surf = int(data_dict["surface_mesh_centers"].shape[0])
        safe_geo = max(1, min(cfg.geometry_sampling, n_stl, n_surf))
        if datapipe is None or geometry_effective != safe_geo:
            datapipe = build_surface_datapipe(
                geometry_sampling=safe_geo,
                surface_factors=surface_factors,
                device=cfg.device,
                user_kw=cfg.datapipe_user_kw,
            )
            geometry_effective = safe_geo

    batch = datapipe(data_dict)
    return TransolverBatch(
        batch=batch,
        datapipe=datapipe,
        geometry_effective=int(safe_geo),
        volume_length_scale=length_scale,
    )


def geotransolver_forward(
    *,
    model: "GeoTransolver",
    batch: dict[str, Any],
    batch_resolution: int,
    cuda_bf16_autocast_enabled: bool,
    device: str,
) -> torch.Tensor:
    """Blocked GeoTransolver forward pass; returns model-space predictions (N, C), order restored."""
    fx_bn_c = global_fx_to_bnc(batch["fx"])
    n = batch["embeddings"].shape[1]
    batch_res = min(batch_resolution, n)
    indices = torch.randperm(n, device=batch["embeddings"].device)
    index_blocks = torch.split(indices, batch_res)
    preds_list: list[torch.Tensor] = []
    use_full_fx = "geometry" in batch

    ac_ctx = (
        cuda_bf16_autocast(device) if cuda_bf16_autocast_enabled else nullcontext()
    )
    with torch.no_grad():
        with ac_ctx:
            for index_block in index_blocks:
                local_embeddings = batch["embeddings"][:, index_block]
                local_fx = fx_bn_c if use_full_fx else fx_bn_c[:, index_block]
                local_positions = local_embeddings[:, :, :3]
                geometry_kw = batch["geometry"] if "geometry" in batch else None
                outputs = model(
                    local_embedding=local_embeddings,
                    local_positions=local_positions,
                    global_embedding=local_fx,
                    geometry=geometry_kw,
                )
                preds_list.append(outputs)
            predictions = torch.cat(preds_list, dim=1)
            inverse_indices = torch.empty_like(indices)
            inverse_indices[indices] = torch.arange(n, device=indices.device)
            predictions = predictions[:, inverse_indices]
    return predictions.squeeze(0)


def unscale_targets(
    *,
    datapipe: "TransolverDataPipe",
    predictions: torch.Tensor,
    batch: dict[str, Any],
    inference_mode: str,
) -> torch.Tensor:
    """Invert the datapipe target scaling (``unscale_model_targets``) for surface/volume."""
    return datapipe.unscale_model_targets(
        predictions,
        air_density=batch.get("air_density"),
        stream_velocity=batch.get("stream_velocity"),
        factor_type="volume" if inference_mode == "volume" else "surface",
    )


def decode_surface_predictions(
    raw_output: RawOutput,
    *,
    redimensionalize: bool,
    air_density: float,
    stream_velocity: float,
) -> Predictions:
    """Map unscaled surface targets to canonical ``pressure`` / ``shear_stress`` (physical units).

    ``redimensionalize`` multiplies by the dynamic pressure ``ρu²`` for non-dimensional-target
    (DrivAerML) checkpoints; it is the identity for physical-target (``transformer_models``) ones.
    """
    pred = raw_output
    if pred.dim() == 3:
        pred = pred.squeeze(0)
    log_inference("geotransolver", "Decoding outputs (pressure + WSS to numpy)…")
    dynamic_pressure = air_density * (stream_velocity**2) if redimensionalize else 1.0
    pressure = (pred[:, 0] * dynamic_pressure).cpu().numpy().astype(np.float32)
    wss = (pred[:, 1:4] * dynamic_pressure).cpu().numpy().astype(np.float32)
    return build_predictions_dict(pressure=pressure, shear_stress=wss)


def decode_volume_predictions(
    raw_output: RawOutput,
    *,
    redimensionalize: bool,
    air_density: float,
    stream_velocity: float,
    length_scale: float | None,
) -> Predictions:
    """Map unscaled volume targets to canonical ``velocity`` / ``pressure`` / ``turbulent_viscosity``.

    ``redimensionalize`` applies the ``u`` (velocity), ``ρu²`` (pressure) and ``u·L`` (νₜ) scales
    for non-dimensional-target checkpoints; identity for physical-target ones.
    """
    pred = raw_output
    if pred.dim() == 3:
        pred = pred.squeeze(0)
    log_inference(
        "geotransolver",
        "Decoding outputs (velocity + pressure + nut → canonical volume keys)…",
    )
    if redimensionalize:
        if length_scale is None:
            raise RuntimeError(
                "Volume decode requires the STL length scale; prepare_inputs must run "
                "before decode_outputs."
            )
        u = float(stream_velocity)
        rho = float(air_density)
        dynamic_pressure = rho * (u**2)
        nut_scale = u * length_scale
    else:
        u = 1.0
        dynamic_pressure = 1.0
        nut_scale = 1.0
    velocity = (pred[:, 0:3] * u).cpu().numpy().astype(np.float32)
    pressure = (pred[:, 3] * dynamic_pressure).cpu().numpy().astype(np.float32)
    turbulent_viscosity = (pred[:, 4] * nut_scale).cpu().numpy().astype(np.float32)
    return build_predictions_dict(
        velocity=velocity,
        pressure=pressure,
        turbulent_viscosity=turbulent_viscosity,
    )


__all__ = [
    "DEFAULT_GEOTRANSOLVER_KW",
    "DEFAULT_GEOTRANSOLVER_VOLUME_KW",
    "DATAPIPE_KEYS",
    "GeoTransolverRuntimeConfig",
    "TransolverBatch",
    "geotransolver_available",
    "global_fx_to_bnc",
    "split_datapipe_kwargs",
    "parse_runtime_kwargs",
    "ensure_distributed_initialized",
    "resolve_checkpoint_file",
    "build_geotransolver_backbone",
    "build_surface_datapipe",
    "build_volume_datapipe",
    "build_transolver_batch",
    "geotransolver_forward",
    "unscale_targets",
    "decode_surface_predictions",
    "decode_volume_predictions",
]
