"""DoMINO model wrapper (surface or volume inference; matches domino ``src/test.py``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.domino.model import DoMINO

from physicsnemo.cfd.evaluation.common.checkpoint_compat import trusted_torch_load_context
from physicsnemo.cfd.evaluation.datasets.schema import CanonicalCase, InferenceDomain, predictions_dict
from physicsnemo.cfd.evaluation.inference.common_wrapper_utils.vtk_datapipe_io import (
    run_id_from_case_id,
)
from physicsnemo.cfd.evaluation.inference.model_registry import (
    CFDModel,
    ModelInput,
    OutputLocation,
    RawOutput,
    Predictions,
)
from physicsnemo.cfd.evaluation.inference.progress import log_inference
from physicsnemo.cfd.evaluation.inference.wrappers.domino.inference import (
    build_domin_surface_datadict,
    build_domin_volume_datadict,
    domino_count_output_features,
    domino_surface_test_step,
    domino_volume_predictions_to_canonical,
    domino_volume_test_step,
)
from physicsnemo.cfd.evaluation.inference.wrappers.domino.scaling import load_scaling_factors_tensors
from physicsnemo.utils import load_checkpoint


class DominoWrapper(CFDModel):
    """DoMINO inference using Hydra-style YAML + checkpoint from domino training.

    **Config (``model.kwargs``)**

    - ``domino_config`` (str): Path to ``config.yaml`` (same schema as domino training / test).
      ``model.model_type`` must be ``surface`` or ``volume`` (``combined`` is not supported here).
    - ``point_batch_size`` (int, optional): Subdomain batch size (default 256000).

    Set ``model.inference_domain: volume`` for VTU cases, or omit it to follow ``model.model_type``
    in the DoMINO YAML.
    """

    INFERENCE_DOMAIN: ClassVar[InferenceDomain] = "surface"
    OUTPUT_LOCATION: ClassVar[OutputLocation] = "cell"

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
        _sf = self._cfg.data.scaling_factors
        if _sf and not Path(str(_sf)).is_absolute():
            OmegaConf.update(
                self._cfg,
                "data.scaling_factors",
                str(_cfg_p.parent / str(_sf)),
            )

        mtype = self._cfg.model.model_type
        if mtype not in ("surface", "volume"):
            raise NotImplementedError(
                "DominoWrapper supports DoMINO config model.model_type "
                f"'surface' or 'volume' only; got {mtype!r} (combined is not supported)."
            )

        dom = kw.pop("inference_domain", None)
        if dom in ("surface", "volume"):
            self._inference_mode = dom
        else:
            self._inference_mode = "volume" if mtype == "volume" else "surface"

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
            pred = domino_surface_test_step(
                model_input["data_dict"],
                self._model,
                model_input["cfg"],
                model_input["surf_factors"],
                dev,
                model_input["point_batch_size"],
            )
        return pred

    def decode_outputs(self, raw_output: RawOutput, case: CanonicalCase) -> Predictions:
        if self._inference_mode == "volume":
            assert self._cfg is not None
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
        return predictions_dict(pressure, wss)
