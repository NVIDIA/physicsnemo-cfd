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

"""DoMINO model wrapper (surface or volume inference; matches domino ``src/test.py``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.domino.model import DoMINO

from physicsnemo.cfd.evaluation.common.checkpoint_compat import parse_checkpoint_epoch, trusted_torch_load_context
from physicsnemo.cfd.evaluation.datasets.schema import CanonicalCase, InferenceDomain, build_predictions_dict, normalize_inference_domain_str
from physicsnemo.cfd.evaluation.models.inference_autocast import cuda_bf16_autocast
from physicsnemo.cfd.evaluation.models.common_wrapper_utils.vtk_datapipe_io import (
    run_id_from_case_id,
)
from physicsnemo.cfd.evaluation.models.model_registry import (
    CFDModel,
    ModelInput,
    OutputLocation,
    RawOutput,
    Predictions,
)
from physicsnemo.cfd.evaluation.inference.progress import log_inference
from physicsnemo.cfd.evaluation.models.wrappers.domino.inference import (
    build_domin_surface_datadict,
    build_domin_volume_datadict,
    domino_count_output_features,
    domino_surface_test_step,
    domino_volume_predictions_to_canonical,
    domino_volume_test_step,
)
from physicsnemo.cfd.evaluation.models.wrappers.domino.scaling import load_scaling_factors_tensors
from physicsnemo.utils import load_checkpoint


class DominoWrapper(CFDModel):
    """DoMINO inference using Hydra-style YAML + checkpoint from domino training.

    **Config (``model.kwargs``)**

    - ``domino_config`` (str): Path to ``config.yaml`` (same schema as domino training / test).
      ``model.model_type`` must be ``surface`` or ``volume`` (``combined`` is not supported here).
    - ``point_batch_size`` (int, optional): Subdomain batch size (default 256000).

    **Benchmark ``model.stats_path``:** When non-empty, overrides the ``data.scaling_factors`` entry in
    ``domino_config`` and any engine-passed ``_resolved_scaling_factors`` (HF package cache path).
    Otherwise, if ``resolve_model_assets`` supplies ``_resolved_scaling_factors``, that wins over stale
    YAML paths; if not, ``data.scaling_factors`` is resolved with relative paths anchored to the YAML
    directory and (for Hub-style layouts) prefers a pickle file colocated with ``domino_config``.
    The pickle must be the training ``ScalingFactors`` artifact, not ``global_stats.json``.

    Set ``model.inference_domain`` for VTU or VTP workloads to match Hydra/domain routing; alternatively
    omit it — :meth:`load` resolves ``surface`` vs ``volume`` from ``domino_config`` (``model.model_type``).
    Routing and metrics use :meth:`inference_domain_from_kwargs` so ``model.model_type: volume``
    aligns with volume datasets without a duplicate YAML field here.
    """

    INFERENCE_DOMAIN: ClassVar[InferenceDomain | None] = None
    OUTPUT_LOCATION: ClassVar[OutputLocation] = "cell"

    @classmethod
    def inference_domain_from_kwargs(cls, kwargs: dict[str, Any]) -> InferenceDomain | None:
        """Match :meth:`load`: read ``model.model_type`` when ``domino_config`` exists and inference_domain omitted."""
        dom_raw = kwargs.get("inference_domain")
        if dom_raw is not None:
            return normalize_inference_domain_str(
                dom_raw,
                parameter="model.kwargs.inference_domain",
            )
        cfg_path = kwargs.get("domino_config") or kwargs.get("config_path")
        if not cfg_path:
            return None
        try:
            dom_cfg = OmegaConf.load(str(cfg_path))
        except Exception:
            return None
        mtype_raw = OmegaConf.select(dom_cfg, "model.model_type")
        if isinstance(mtype_raw, str):
            return normalize_inference_domain_str(
                mtype_raw,
                parameter="Domino YAML model.model_type",
            )
        return None

    @property
    def output_location(self) -> OutputLocation:
        return self.OUTPUT_LOCATION

    def __init__(self) -> None:
        self._model: Optional[DoMINO] = None
        self._cfg: Optional[DictConfig] = None
        self._surf_factors: Optional[torch.Tensor] = None
        self._vol_factors: Optional[torch.Tensor] = None
        self._device: str = "cuda:0"
        self._point_batch_size: int = 256_000
        self._inference_mode: Literal["surface", "volume"] = "surface"

    def load(
        self,
        checkpoint_path: str,
        stats_path: str,
        device: str,
        **kwargs: Any,
    ) -> "DominoWrapper":
        kw = dict(kwargs)
        # Exact HF-cache path from ``resolve_model_assets`` (overrides stale paths inside shipped ``config.yaml``).
        _resolved_sf = kw.pop("_resolved_scaling_factors", None)
        cfg_path = kw.get("domino_config") or kw.get("config_path")
        if not cfg_path:
            raise ValueError(
                "DominoWrapper requires model.kwargs.domino_config (path to DoMINO config.yaml)."
            )
        self._device = device
        self._point_batch_size = int(kw.get("point_batch_size", 256_000))

        if not DistributedManager.is_initialized():
            DistributedManager.initialize()

        log_inference(
            "domino",
            f"Loading DoMINO config from {cfg_path}; checkpoint from {checkpoint_path}",
        )
        self._cfg = OmegaConf.load(cfg_path)
        _cfg_p = Path(cfg_path).resolve()
        _cfg_dir = _cfg_p.parent
        stats_override = (stats_path or "").strip()
        if stats_override:
            raw = Path(stats_override).expanduser()
            pickle_path = raw if raw.is_absolute() else (_cfg_dir / raw).resolve()
            if not pickle_path.is_file():
                if pickle_path.is_dir():
                    raise ValueError(
                        "DoMINO scaling path must be a file (training ScalingFactors pickle), "
                        f"not a directory: {pickle_path!r}. Set benchmark model.stats_path to that pickle, "
                        "or leave stats_path empty to use domino_config data.scaling_factors."
                    )
                raise FileNotFoundError(
                    "DoMINO scaling factors pickle not found (benchmark model.stats_path override): "
                    f"{pickle_path}"
                )
            # Other benchmark models use ``global_stats.json``; DoMINO expects ``scaling_factors.pkl``.
            if pickle_path.suffix.lower() == ".json":
                sibling_pkl = (pickle_path.parent / "scaling_factors.pkl").resolve()
                if sibling_pkl.is_file():
                    log_inference(
                        "domino",
                        "stats_path points at JSON (e.g. global_stats.json); "
                        f"using DoMINO pickle {sibling_pkl}",
                    )
                    pickle_path = sibling_pkl
                else:
                    raise ValueError(
                        "DoMINO normalization is loaded from a training ScalingFactors pickle "
                        "(``scaling_factors.pkl``), not ``global_stats.json``. "
                        f"No scaling_factors.pkl next to {pickle_path}. "
                        "Set model.stats_path to that pickle, or place scaling_factors.pkl beside the JSON."
                    )
            OmegaConf.update(self._cfg, "data.scaling_factors", str(pickle_path))
            log_inference(
                "domino",
                f"Using benchmark model.stats_path as data.scaling_factors override: {pickle_path}",
            )
        elif _resolved_sf:
            OmegaConf.update(
                self._cfg,
                "data.scaling_factors",
                str(Path(_resolved_sf).resolve()),
            )
        else:
            _sf = self._cfg.data.scaling_factors
            if _sf:
                _sf_path = Path(str(_sf)).expanduser()
                _pickle_name = _sf_path.name or "scaling_factors.pkl"
                _local_by_config = _cfg_dir / _pickle_name
                # Prefer pickle next to ``config.yaml`` (Hub layout) over stale absolute paths in YAML.
                if _local_by_config.is_file():
                    OmegaConf.update(
                        self._cfg,
                        "data.scaling_factors",
                        str(_local_by_config.resolve()),
                    )
                elif not _sf_path.is_absolute():
                    OmegaConf.update(
                        self._cfg,
                        "data.scaling_factors",
                        str(_cfg_dir / str(_sf)),
                    )
                elif not _sf_path.exists():
                    _fallback = _cfg_dir / _pickle_name
                    if _fallback.is_file():
                        OmegaConf.update(
                            self._cfg,
                            "data.scaling_factors",
                            str(_fallback.resolve()),
                        )

        try:
            mtype = normalize_inference_domain_str(
                str(self._cfg.model.model_type),
                parameter="Domino YAML model.model_type",
            )
        except ValueError:
            raise NotImplementedError(
                "DominoWrapper supports DoMINO config model.model_type "
                f"'surface' or 'volume' only; got {self._cfg.model.model_type!r} (combined is not supported)."
            ) from None

        dom = kw.pop("inference_domain", None)
        if dom is None:
            self._inference_mode = "volume" if mtype == "volume" else "surface"
        else:
            self._inference_mode = normalize_inference_domain_str(
                dom,
                parameter="model.kwargs.inference_domain",
            )

        if self._inference_mode != mtype:
            raise ValueError(
                f"model.inference_domain is {self._inference_mode!r} but DoMINO config "
                f"model.model_type is {mtype!r}; they must match."
            )

        dev = torch.device(device)
        vol_f, surf_f = load_scaling_factors_tensors(self._cfg, dev)
        self._vol_factors = vol_f
        self._surf_factors = surf_f
        if self._inference_mode == "surface":
            if self._surf_factors is None:
                raise RuntimeError("Surface scaling factors missing.")
        else:
            if self._vol_factors is None:
                raise RuntimeError("Volume scaling factors missing.")

        num_vol, num_surf, num_glob = domino_count_output_features(self._cfg)
        self._model = DoMINO(
            input_features=3,
            output_features_vol=num_vol,
            output_features_surf=num_surf,
            global_features=num_glob,
            model_parameters=self._cfg.model,
        ).to(dev)

        ckpt = Path(checkpoint_path)
        # ``physicsnemo.utils.load_checkpoint`` expects a checkpoint *directory*, not a .pt path.
        checkpoint_dir = ckpt.parent if ckpt.is_file() else ckpt
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"Checkpoint path must be a directory or a file inside one; got {checkpoint_path!r}"
            )
        log_inference("domino", f"Loading checkpoint from directory {checkpoint_dir}")

        ckpt_args = {
            "path": str(checkpoint_dir),
            "models": self._model,
        }
        epoch = parse_checkpoint_epoch(checkpoint_path)
        if epoch is not None:
            ckpt_args["epoch"] = epoch

        with trusted_torch_load_context():
            loaded_epoch = load_checkpoint(device=dev, **ckpt_args)
        self._model.eval()
        log_inference("domino", "Checkpoint loaded; model ready for inference.")
        return self

    def prepare_inputs(self, case: CanonicalCase) -> ModelInput:
        if self._model is None or self._cfg is None:
            raise RuntimeError("DominoWrapper: call load() first")
        log_inference(
            "domino",
            f"Reading case inputs (case {case.case_id}): mesh {case.mesh_path}, "
            f"run dir {Path(case.mesh_path).parent}",
        )
        run_dir = Path(case.mesh_path).parent
        tag = run_id_from_case_id(case.case_id)
        dev = torch.device(self._device)

        if self._inference_mode == "volume":
            data_dict = build_domin_volume_datadict(
                self._cfg, run_dir, case.mesh_path, tag, dev
            )
            return {
                "data_dict": data_dict,
                "cfg": self._cfg,
                "vol_factors": self._vol_factors,
                "point_batch_size": self._point_batch_size,
                "mode": "volume",
            }

        data_dict = build_domin_surface_datadict(
            self._cfg, run_dir, case.mesh_path, tag, dev
        )
        return {
            "data_dict": data_dict,
            "cfg": self._cfg,
            "surf_factors": self._surf_factors,
            "point_batch_size": self._point_batch_size,
            "mode": "surface",
        }

    def predict(self, model_input: ModelInput) -> RawOutput:
        if self._model is None:
            raise RuntimeError("DominoWrapper: call load() first")
        dev = torch.device(self._device)
        if model_input.get("mode") == "volume":
            log_inference("domino", "Running forward pass (predicting volume fields)…")
            with torch.no_grad():
                with cuda_bf16_autocast(dev):
                    pred = domino_volume_test_step(
                        model_input["data_dict"],
                        self._model,
                        model_input["cfg"],
                        model_input["vol_factors"],
                        dev,
                        model_input["point_batch_size"],
                    )
            return pred

        log_inference("domino", "Running forward pass (predicting surface fields)…")
        with torch.no_grad():
            with cuda_bf16_autocast(dev):
                pred = domino_surface_test_step(
                    model_input["data_dict"],
                    self._model,
                    model_input["cfg"],
                    model_input["surf_factors"],
                    dev,
                    model_input["point_batch_size"],
                )
        return pred

    def decode_outputs(
        self,
        raw_output: RawOutput,
        case: CanonicalCase,
        model_input: Optional[ModelInput] = None,
    ) -> Predictions:
        if self._inference_mode == "volume":
            if self._cfg is None:
                raise RuntimeError("DominoWrapper: call load() first")
            log_inference(
                "domino",
                "Decoding volume outputs (canonical keys from variables.volume.solution)…",
            )
            return domino_volume_predictions_to_canonical(raw_output, self._cfg)

        log_inference("domino", "Decoding outputs (pressure + WSS to numpy)…")
        pred = raw_output
        if pred.dim() == 3:
            pred = pred.squeeze(0)
        pressure = pred[:, 0].cpu().numpy().astype("float32")
        wss = pred[:, 1:4].cpu().numpy().astype("float32")
        return build_predictions_dict(pressure=pressure, shear_stress=wss)
