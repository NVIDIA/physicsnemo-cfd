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

"""DrivAerML dataset adapter: surface VTP and/or volume VTU under each ``run_*`` directory."""

from pathlib import Path
from typing import Any

import pyvista as pv

from physicsnemo.cfd.evaluation.datasets.adapter_registry import DatasetAdapter
from physicsnemo.cfd.evaluation.datasets.schema import CanonicalCase, InferenceDomain, coerce_inference_domain_or_default
from physicsnemo.cfd.evaluation.common.natural_sort import natural_sorted
from physicsnemo.cfd.evaluation.datasets.progress import log_dataset
from physicsnemo.cfd.evaluation.datasets.vtk_ground_truth import (
    DEFAULT_PRESSURE_NAMES,
    DEFAULT_SHEAR_NAMES,
    extract_pressure_wss_from_mesh,
    extract_volume_fields_from_mesh,
)


#: DrivAer/OpenFOAM volume VTUs commonly expose ``*MeanTrim`` (truncated-domain time averages).
#: The adapter prepends these to the generic CFD names so default DrivAer runs find velocity
#: and νₜ without requiring per-config ``dataset.kwargs.velocity_field_names`` overrides.
DRIVAER_VOLUME_VELOCITY_NAMES: tuple[str, ...] = ("UMeanTrim", "UMean", "U", "velocity", "Velocity")
DRIVAER_TURBULENT_VISCOSITY_NAMES: tuple[str, ...] = (
    "nutMeanTrim",
    "nutMean",
    "nut",
    "turbulent_viscosity",
    "TurbulentViscosity",
    "nuTilde",
)


class DrivAerMLAdapter(DatasetAdapter):
    """Adapter for DrivAerML layout under ``root/run_<id>/``.

    **Surface (default)** — boundary VTP per run aligns with the run tag, same idea as volume:
    ``run_<n>/boundary_<n>.vtp`` (e.g. ``run_1`` → ``boundary_1.vtp``). Override with
    ``boundary_vtp_filename`` (fixed name in every run dir) or ``boundary_vtp_template``
    (e.g. ``"boundary_{run_suffix}.vtp"``; supports ``{run_suffix}``, ``{case_id}``).

    Benchmarks use **all** ``run_*`` directories under ``dataset.root`` that contain the
    required mesh (or the subset in ``dataset.case_ids``). Published
    train/validation CSVs in the workflow folder are not applied by the benchmark driver.

    **Volume** — set ``dataset.kwargs.inference_domain: volume``. Default
    ``run_<n>/volume_<n>.vtu``. Override with ``volume_vtu_filename`` or ``volume_vtu_template``.

    Optional kwargs (from ``dataset.kwargs`` in config):

    - ``inference_domain``: ``"surface"`` (default) or ``volume``.
    - ``gt_data_type``: for surface, ``auto`` / ``cell`` / ``point`` / ``from_model``
      (below). Surface ``auto`` and ``cell`` are equivalent (both try **cell** then **point**).
      For volume, passed to volume GT extraction as ``auto`` / ``cell`` / ``point`` only
      (``auto`` / ``point`` prefer **point** then **cell**; ``cell`` prefers **cell** first).
    - ``align_ground_truth_to_model``: surface only; same as elsewhere.
    - ``pressure_field_names``, ``shear_field_names``: surface GT array name overrides.
    - ``turbulent_viscosity_field_names``, ``velocity_field_names``,
      ``volume_pressure_field_names``: volume GT VTK array name tuples. Adapter defaults
      (``DRIVAER_VOLUME_VELOCITY_NAMES`` / ``DRIVAER_TURBULENT_VISCOSITY_NAMES``) prepend the
      DrivAer/OpenFOAM ``*MeanTrim`` variant so DrivAer VTUs work without overrides. Volume
      pressure falls back to ``DEFAULT_VOLUME_PRESSURE_NAMES`` (already includes ``pMeanTrim``).
    - ``boundary_vtp_filename``: if set, every run uses this exact VTP name (legacy e.g. ``boundary_1.vtp``).
    - ``boundary_vtp_template``: custom surface filename pattern.
    - ``volume_vtu_filename``: if set, every run uses this exact VTU name inside the run dir.
    - ``volume_vtu_template``: format string with ``{run_suffix}`` (default pattern above).

    Deprecated alias: ``gt_prefer`` for ``gt_data_type``.
    """

    @classmethod
    def inference_domain(cls) -> InferenceDomain:
        return "surface"

    @classmethod
    def inference_domain_from_kwargs(cls, kwargs: dict[str, Any] | None) -> InferenceDomain:
        kw = kwargs or {}
        raw = kw.get("inference_domain")
        return coerce_inference_domain_or_default(
            raw,
            default="surface",
            parameter="dataset.kwargs.inference_domain",
        )

    def __init__(self, root: str, **kwargs: Any) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"DrivAerML root not found: {self.root}")
        self._inference_mode: InferenceDomain = self.inference_domain_from_kwargs(kwargs)

        self._gt_data_type = kwargs.get("gt_data_type", kwargs.get("gt_prefer", "auto"))
        pn = kwargs.get("pressure_field_names")
        sn = kwargs.get("shear_field_names")
        self._pressure_names: tuple[str, ...] = (
            tuple(pn) if pn else DEFAULT_PRESSURE_NAMES
        )
        self._shear_names: tuple[str, ...] = tuple(sn) if sn else DEFAULT_SHEAR_NAMES

        self._boundary_vtp_filename: str | None = kwargs.get("boundary_vtp_filename")
        self._boundary_vtp_template: str | None = kwargs.get("boundary_vtp_template")

        self._volume_vtu_filename: str | None = kwargs.get("volume_vtu_filename")
        self._volume_vtu_template: str | None = kwargs.get("volume_vtu_template")

        tn = kwargs.get("turbulent_viscosity_field_names")
        vn = kwargs.get("velocity_field_names")
        pn_vol = kwargs.get("volume_pressure_field_names")
        self._nut_names: tuple[str, ...] = (
            tuple(tn) if tn else DRIVAER_TURBULENT_VISCOSITY_NAMES
        )
        self._vel_names: tuple[str, ...] = (
            tuple(vn) if vn else DRIVAER_VOLUME_VELOCITY_NAMES
        )
        self._volume_pressure_names: tuple[str, ...] = tuple(pn_vol) if pn_vol else ()

    def _run_suffix(self, case_id: str) -> str:
        return case_id[4:] if case_id.startswith("run_") else case_id

    def _boundary_vtp_path(self, run_dir: Path, case_id: str) -> Path:
        if self._boundary_vtp_filename:
            return run_dir / self._boundary_vtp_filename
        if self._boundary_vtp_template:
            return run_dir / self._boundary_vtp_template.format(
                run_suffix=self._run_suffix(case_id),
                case_id=case_id,
            )
        return run_dir / f"boundary_{self._run_suffix(case_id)}.vtp"

    def _volume_vtu_path(self, run_dir: Path, case_id: str) -> Path:
        if self._volume_vtu_filename:
            return run_dir / self._volume_vtu_filename
        if self._volume_vtu_template:
            return run_dir / self._volume_vtu_template.format(
                run_suffix=self._run_suffix(case_id),
                case_id=case_id,
            )
        return run_dir / f"volume_{self._run_suffix(case_id)}.vtu"

    def list_cases(self) -> list[str]:
        """Return case IDs: run directory names that contain the required mesh for the mode."""
        case_ids: list[str] = []
        for p in self.root.iterdir():
            if not p.is_dir() or not p.name.startswith("run_"):
                continue
            if self._inference_mode == "volume":
                if self._volume_vtu_path(p, p.name).exists():
                    case_ids.append(p.name)
            else:
                if self._boundary_vtp_path(p, p.name).exists():
                    case_ids.append(p.name)
        return natural_sorted(case_ids)

    def load_case(self, case_id: str) -> CanonicalCase:
        """Load surface VTP or volume VTU and optional ground truth."""
        log_dataset(
            "drivaerml",
            f"load_case({case_id!r}): branch={self._inference_mode!r}, root={self.root}",
        )
        run_dir = self.root / case_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Case not found: {run_dir}")

        if self._inference_mode == "volume":
            return self._load_volume_case(case_id, run_dir)
        return self._load_surface_case(case_id, run_dir)

    def _load_surface_case(self, case_id: str, run_dir: Path) -> CanonicalCase:
        mesh_path = self._boundary_vtp_path(run_dir, case_id)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Surface mesh not found: {mesh_path}")

        log_dataset("drivaerml", f"Reading surface mesh from {mesh_path}")
        mesh = pv.read(str(mesh_path))
        if not isinstance(mesh, pv.PolyData):
            mesh = mesh.extract_surface()

        gt_dict, gt_loc = extract_pressure_wss_from_mesh(
            mesh,
            data_type=self._gt_data_type,
            pressure_names=self._pressure_names,
            shear_names=self._shear_names,
        )
        ground_truth = gt_dict if gt_dict else None
        mesh_type = gt_loc if gt_loc is not None else "unknown"

        meta: dict[str, Any] = {
            "dataset": "drivaerml",
            "run": case_id,
            "branch": "surface",
            "boundary_vtp": mesh_path.name,
        }
        if ground_truth:
            meta["ground_truth_location"] = gt_loc
            meta["ground_truth_fields"] = list(ground_truth.keys())

        return CanonicalCase(
            case_id=case_id,
            mesh_path=str(mesh_path),
            mesh_type=mesh_type,
            ground_truth=ground_truth,
            metadata=meta,
            inference_domain="surface",
            reference_geometry=mesh,
        )

    def _load_volume_case(self, case_id: str, run_dir: Path) -> CanonicalCase:
        mesh_path = self._volume_vtu_path(run_dir, case_id)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Volume mesh not found: {mesh_path}")

        log_dataset("drivaerml", f"Reading volume mesh from {mesh_path}")
        mesh = pv.read(str(mesh_path))
        ugrid = (
            mesh.cast_to_unstructured_grid()
            if hasattr(mesh, "cast_to_unstructured_grid")
            else mesh
        )

        v_gt_type = self._gt_data_type
        if v_gt_type not in ("auto", "cell", "point"):
            v_gt_type = "auto"

        gt_dict, gt_loc = extract_volume_fields_from_mesh(
            ugrid,
            data_type=v_gt_type,
            turbulent_viscosity_names=self._nut_names if self._nut_names else None,
            velocity_names=self._vel_names if self._vel_names else None,
            pressure_names=self._volume_pressure_names if self._volume_pressure_names else None,
        )
        ground_truth = gt_dict if gt_dict else None
        mesh_type = gt_loc if gt_loc is not None else "point"  # nodal VTU default; matches mesh_bridge volume

        meta: dict[str, Any] = {
            "dataset": "drivaerml",
            "run": case_id,
            "branch": "volume",
            "volume_vtu": mesh_path.name,
        }
        if ground_truth:
            meta["ground_truth_location"] = gt_loc
            meta["ground_truth_fields"] = list(ground_truth.keys())

        return CanonicalCase(
            case_id=case_id,
            mesh_path=str(mesh_path.resolve()),
            mesh_type=mesh_type,
            ground_truth=ground_truth,
            metadata=meta,
            inference_domain="volume",
            reference_geometry=ugrid,
        )
