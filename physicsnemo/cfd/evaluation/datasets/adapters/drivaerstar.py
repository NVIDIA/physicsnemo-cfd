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

"""DrivAerStar dataset adapter: flat directory of legacy surface ``.vtk`` files.

`DrivAerStar <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/UXVXQV>`_
is an industrial-grade automotive CFD dataset whose surface meshes ship as legacy
``.vtk`` PolyData in a flat directory (``<case_id>.vtk``), with field names and a wall
shear stress sign convention that differ from DrivAerML (which the shipped GeoTransolver
/ Transolver checkpoints were trained on). This adapter bridges those differences so the
same model wrappers and metrics work unchanged:

1. Rename ``Pressure`` → ``pMeanTrim`` (DrivAerML convention).
2. Combine the three WSS scalar components into one ``(N, 3)`` vector and flip its sign
   to match DrivAerML (``flip_wss_sign``, default on).
3. Drop explicit ``Normals`` / ``Area`` arrays so downstream force integration and
   rendering recompute them from mesh topology (DrivAerML convention).

These transforms are cheap generic array ops, so by default they run **in memory** on each pass:
the prepared surface mesh is handed to the wrappers via
:attr:`~physicsnemo.cfd.evaluation.datasets.schema.CanonicalCase.reference_geometry` (predictions +
ground truth) and :attr:`~physicsnemo.cfd.evaluation.datasets.schema.CanonicalCase.geometry` (the
SDF / geometry branch — the DrivAerStar surface *is* its geometry, so no STL file is materialized).
Nothing is written to disk, avoiding a full-dataset duplicate and any stale-cache class of bug.
``mesh_path`` points at the source ``.vtk`` (used only for run-index / directory context; the
forward pass reads the in-memory meshes, not this file).

Set ``cache_prepared: true`` to additionally **persist** a canonical ``.vtp`` (+ triangulated
``.stl``) under ``<root>/<prepared_subdir>/<case_id>/`` for external inspection (ParaView, etc.).
Those writes are guarded by a sidecar signature (``<case_id>.prepared.json``) recording the
transform kwargs (``flip_wss_sign``, field names, ``remove_normals_area``, ...) and the source
file's identity (mtime/size), so a changed transform rebuilds rather than reusing a stale artifact.

.. note::
   The datapipe wrappers (GeoTransolver, Transolver) parse an integer run index from the
   case id via ``run_id_from_case_id``, so case ids must be integer-like (a decimal string or
   ``run_<int>``). DrivAerStar VTK stems are integers, so this holds out of the box. Rename files
   or pass ``glob_pattern`` accordingly if yours are not.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from physicsnemo.cfd.evaluation.common.natural_sort import natural_sorted
from physicsnemo.cfd.evaluation.datasets.adapter_registry import DatasetAdapter
from physicsnemo.cfd.evaluation.datasets.progress import log_dataset
from physicsnemo.cfd.evaluation.datasets.schema import (
    CanonicalCase,
    InferenceDomain,
    coerce_inference_domain_or_default,
)
from physicsnemo.cfd.evaluation.datasets.vtk_ground_truth import (
    extract_pressure_wss_from_mesh,
)

#: DrivAerStar source array names (legacy ``.vtk`` cell data).
DEFAULT_SOURCE_PRESSURE_NAME = "Pressure"
DEFAULT_SOURCE_WSS_COMPONENT_NAMES: tuple[str, str, str] = (
    "WallShearStressi",
    "WallShearStressj",
    "WallShearStressk",
)

#: Canonical (DrivAerML) VTK array names written into the prepared VTP. Chosen so the
#: default ``output.ground_truth_mesh_field_names`` (which target DrivAerML names) work
#: without per-config overrides.
DEFAULT_PRESSURE_OUT_NAME = "pMeanTrim"
DEFAULT_SHEAR_OUT_NAME = "wallShearStressMeanTrim"


def _run_id_from_case_id(case_id: str) -> int:
    """Parse an integer run index from ``run_<n>`` or a decimal id (mirrors the datapipe
    wrappers' ``run_id_from_case_id`` so the STL name resolves to the same tag)."""
    s = str(case_id).strip()
    if not s:
        raise ValueError("case_id is empty")
    suffix = s[4:] if s.startswith("run_") else s
    return int(suffix)


class DrivAerStarAdapter(DatasetAdapter):
    """Adapter for DrivAerStar surface ``.vtk`` files in a flat directory.

    Case ids are the VTK file stems. Ground truth is exposed under the canonical keys
    ``pressure`` (N,) and ``shear_stress`` (N, 3).

    Optional kwargs (from ``dataset.kwargs`` in config):

    - ``inference_domain``: ``"surface"`` (default). Volume is not supported (DrivAerStar
      surface meshes only).
    - ``glob_pattern``: source-file glob relative to ``root`` (default ``"*.vtk"``).
      Supports ``**`` for nested layouts (case id remains the file stem).
    - ``pressure_field_name``: source pressure array name (default ``"Pressure"``).
    - ``wss_component_names``: three source WSS scalar names (default
      ``("WallShearStressi", "WallShearStressj", "WallShearStressk")``).
    - ``flip_wss_sign``: wall-shear-stress **sign convention** knob. When ``True`` (default) the
      combined WSS is negated so the DrivAerStar ground truth matches the DrivAerML sign
      convention — use this for DrivAerML-convention checkpoints. Set it ``False`` for checkpoints
      trained directly on the **native** DrivAerStar WSS sign (e.g. the GeoTransolver
      ``transformer_models`` checkpoints scored in the UQ example config, which is why that config
      sets ``flip_wss_sign: false``). This MUST match how your checkpoints were trained; a mismatch
      silently flips WSS — and therefore drag and lift — for every case.
    - ``remove_normals_area``: drop explicit ``Normals`` / ``Area`` arrays (default ``True``).
    - ``gt_data_type``: ``auto`` / ``cell`` / ``point`` passed to GT extraction
      (default ``"cell"``; DrivAerStar fields are cell-centered).
    - ``pressure_out_name`` / ``shear_out_name``: canonical VTK array names for the prepared
      pressure / WSS arrays (defaults ``"pMeanTrim"`` / ``"wallShearStressMeanTrim"``).
    - ``cache_prepared``: also persist the prepared ``.vtp`` (+ triangulated ``.stl``) to disk for
      external inspection (default ``False`` — preparation is in-memory only).
    - ``prepared_subdir``: cache directory name under ``root`` when ``cache_prepared`` (default
      ``"_prepared"``).
    - ``force_reprepare``: rebuild the on-disk cache even if present, when ``cache_prepared``
      (default ``False``).
    """

    @classmethod
    def inference_domain(cls) -> InferenceDomain:
        """Class-level default inference domain (DrivAerStar is surface-only)."""
        return "surface"

    @classmethod
    def inference_domain_from_kwargs(
        cls, kwargs: dict[str, Any] | None
    ) -> InferenceDomain:
        """Resolve inference domain from ``dataset.kwargs`` (surface only)."""
        kw = kwargs or {}
        domain = coerce_inference_domain_or_default(
            kw.get("inference_domain"),
            default="surface",
            parameter="dataset.kwargs.inference_domain",
        )
        if domain != "surface":
            raise NotImplementedError(
                "DrivAerStarAdapter supports surface inference only; got "
                f"inference_domain={domain!r}."
            )
        return domain

    def __init__(self, root: str, **kwargs: Any) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"DrivAerStar root not found: {self.root}")
        # Validate / normalize domain (raises for volume).
        self.inference_domain_from_kwargs(kwargs)

        self._glob_pattern: str = kwargs.get("glob_pattern", "*.vtk")
        self._pressure_field_name: str = kwargs.get(
            "pressure_field_name", DEFAULT_SOURCE_PRESSURE_NAME
        )
        wss = kwargs.get("wss_component_names", DEFAULT_SOURCE_WSS_COMPONENT_NAMES)
        self._wss_component_names: tuple[str, ...] = tuple(wss)
        if len(self._wss_component_names) != 3:
            raise ValueError(
                "wss_component_names must have exactly 3 entries; got "
                f"{self._wss_component_names!r}"
            )

        self._flip_wss_sign: bool = bool(kwargs.get("flip_wss_sign", True))
        self._remove_normals_area: bool = bool(kwargs.get("remove_normals_area", True))
        self._gt_data_type: str = kwargs.get("gt_data_type", "cell")
        self._cache_prepared: bool = bool(kwargs.get("cache_prepared", False))
        self._prepared_subdir: str = kwargs.get("prepared_subdir", "_prepared")
        self._pressure_out_name: str = kwargs.get(
            "pressure_out_name", DEFAULT_PRESSURE_OUT_NAME
        )
        self._shear_out_name: str = kwargs.get("shear_out_name", DEFAULT_SHEAR_OUT_NAME)
        self._force_reprepare: bool = bool(kwargs.get("force_reprepare", False))

        self._prepared_root = self.root / self._prepared_subdir

    def _source_path(self, case_id: str) -> Path:
        """Resolve the source ``.vtk`` for a case id (stem), honoring ``glob_pattern`` nesting."""
        suffix = Path(self._glob_pattern).suffix or ".vtk"
        direct = self.root / f"{case_id}{suffix}"
        if direct.exists():
            return direct
        for candidate in self.root.glob(self._glob_pattern):
            if candidate.stem == case_id:
                return candidate
        raise FileNotFoundError(
            f"Source mesh for case {case_id!r} not found under {self.root} "
            f"(glob_pattern={self._glob_pattern!r})"
        )

    def _stl_tag(self, case_id: str) -> str:
        """STL tag matching the wrappers' ``run_id_from_case_id`` when derivable."""
        try:
            return str(_run_id_from_case_id(case_id))
        except ValueError:
            # Non-integer id: the datapipe wrappers can't use this case, but keep a stable
            # STL name so GT-only / other consumers still work (one STL per case dir).
            return case_id

    def list_cases(self) -> list[str]:
        """Return case ids: stems of source ``.vtk`` files under ``root`` (excluding cache)."""
        prepared = self._prepared_root.resolve()
        case_ids: list[str] = []
        for p in self.root.glob(self._glob_pattern):
            if not p.is_file():
                continue
            if prepared in p.resolve().parents:
                continue
            case_ids.append(p.stem)
        return natural_sorted(case_ids)

    def _transform_source_mesh(self, case_id: str) -> tuple[pv.PolyData, Path]:
        """Read the raw ``.vtk`` and apply the in-memory canonicalization transforms.

        Shared core of both the default (in-memory) and the opt-in cached paths: rename pressure,
        combine + optionally sign-flip WSS, and drop ``Normals`` / ``Area``. No disk writes. Returns
        the prepared surface mesh and the source path.
        """
        src_path = self._source_path(case_id)
        mesh = pv.read(str(src_path))
        if not isinstance(mesh, pv.PolyData):
            mesh = mesh.extract_surface()
        self._rename_pressure(mesh, case_id)
        self._combine_wss(mesh, case_id)
        if self._remove_normals_area:
            self._drop_normals_area(mesh)
        return mesh, src_path

    #: Bump when the preparation transform changes in a way that invalidates cached artifacts.
    _PREPARE_SIGNATURE_VERSION = 1

    def _transform_signature(self, src_path: Path) -> dict[str, Any]:
        """Signature of everything that affects the prepared artifacts (transform kwargs + source id).

        A cached VTP/STL is only reused when this matches the sidecar written when it was prepared,
        so changing e.g. ``flip_wss_sign`` or a field name — or editing the source ``.vtk`` —
        forces a rebuild instead of a stale cache hit.
        """
        st = src_path.stat()
        return {
            "version": self._PREPARE_SIGNATURE_VERSION,
            "flip_wss_sign": self._flip_wss_sign,
            "remove_normals_area": self._remove_normals_area,
            "pressure_field_name": self._pressure_field_name,
            "wss_component_names": list(self._wss_component_names),
            "pressure_out_name": self._pressure_out_name,
            "shear_out_name": self._shear_out_name,
            "source_name": src_path.name,
            "source_mtime_ns": st.st_mtime_ns,
            "source_size": st.st_size,
        }

    @staticmethod
    def _read_signature(sidecar_path: Path) -> dict[str, Any] | None:
        """Load a prepared-case sidecar; treat a missing/corrupt file as no signature."""
        try:
            with sidecar_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def _write_prepared_cache(
        self, case_id: str, mesh: pv.PolyData, src_path: Path
    ) -> str:
        """Persist the prepared ``.vtp`` (+ triangulated ``.stl``) for external inspection.

        Only called when ``cache_prepared`` is set. Writes are guarded by the sidecar signature so a
        changed transform (or edited source) rebuilds rather than reusing a stale artifact. Returns
        the prepared VTP path (used as the case ``mesh_path`` in cached mode).
        """
        case_dir = self._prepared_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        vtp_path = case_dir / f"{case_id}.vtp"
        stl_path = case_dir / f"drivaer_{self._stl_tag(case_id)}.stl"
        sidecar_path = case_dir / f"{case_id}.prepared.json"

        signature = self._transform_signature(src_path)
        cache_valid = (
            not self._force_reprepare
            and vtp_path.exists()
            and stl_path.exists()
            and self._read_signature(sidecar_path) == signature
        )
        if not cache_valid:
            log_dataset("drivaerstar", f"Caching prepared VTP + STL for {case_id!r}")
            mesh.save(str(vtp_path))
            mesh.triangulate().save(str(stl_path))
            # Write the sidecar only after successful saves so a crash mid-write does not leave a
            # signature that would validate partial/absent artifacts.
            with sidecar_path.open("w", encoding="utf-8") as fh:
                json.dump(signature, fh, indent=2, sort_keys=True)
        return str(vtp_path)

    @staticmethod
    def _available_arrays(mesh: pv.PolyData) -> str:
        """Human-readable listing of cell/point array names for error messages."""
        return (
            f"cell_data={sorted(mesh.cell_data.keys())}, "
            f"point_data={sorted(mesh.point_data.keys())}"
        )

    def _rename_pressure(self, mesh: pv.PolyData, case_id: str) -> None:
        for data in (mesh.cell_data, mesh.point_data):
            if self._pressure_field_name in data:
                data[self._pressure_out_name] = data.pop(self._pressure_field_name)
                return
        raise ValueError(
            f"DrivAerStar case {case_id!r}: pressure array "
            f"{self._pressure_field_name!r} not found in the source mesh "
            f"(check ``pressure_field_name``). Available arrays: "
            f"{self._available_arrays(mesh)}."
        )

    def _combine_wss(self, mesh: pv.PolyData, case_id: str) -> None:
        for data in (mesh.cell_data, mesh.point_data):
            if all(k in data for k in self._wss_component_names):
                wss = np.stack(
                    [np.asarray(data.pop(k)) for k in self._wss_component_names],
                    axis=1,
                ).astype(np.float32)
                if self._flip_wss_sign:
                    wss = -wss
                data[self._shear_out_name] = wss
                return
        raise ValueError(
            f"DrivAerStar case {case_id!r}: wall-shear-stress components "
            f"{list(self._wss_component_names)!r} not all found in a single data group "
            f"(check ``wss_component_names``). Available arrays: "
            f"{self._available_arrays(mesh)}."
        )

    @staticmethod
    def _drop_normals_area(mesh: pv.PolyData) -> None:
        for key in ("Normals", "Area"):
            if key in mesh.cell_data:
                del mesh.cell_data[key]
            if key in mesh.point_data:
                del mesh.point_data[key]

    def load_case(self, case_id: str) -> CanonicalCase:
        """Transform the case in memory (optionally caching) and load it into the canonical schema.

        The prepared surface mesh is returned as both ``reference_geometry`` (predictions + GT) and
        ``geometry`` (the wrappers' SDF branch — surface *is* geometry for DrivAerStar), so the
        forward pass needs no on-disk VTP/STL. ``mesh_path`` is the source ``.vtk`` unless
        ``cache_prepared`` persisted a ``.vtp`` (then it points there).
        """
        log_dataset(
            "drivaerstar",
            f"load_case({case_id!r}): root={self.root}",
        )
        mesh, src_path = self._transform_source_mesh(case_id)
        if self._cache_prepared:
            mesh_path = self._write_prepared_cache(case_id, mesh, src_path)
        else:
            mesh_path = str(src_path)

        gt_dict, gt_loc = extract_pressure_wss_from_mesh(
            mesh,
            data_type=self._gt_data_type,
            pressure_names=(self._pressure_out_name,),
            shear_names=(self._shear_out_name,),
        )
        ground_truth = gt_dict if gt_dict else None
        mesh_type = gt_loc if gt_loc is not None else "cell"

        meta: dict[str, Any] = {
            "dataset": "drivaerstar",
            "case": case_id,
            "branch": "surface",
            "source_vtk": src_path.name,
            "prepared_cached": self._cache_prepared,
        }
        if ground_truth:
            meta["ground_truth_location"] = gt_loc
            meta["ground_truth_fields"] = list(ground_truth.keys())

        return CanonicalCase(
            case_id=case_id,
            mesh_path=mesh_path,
            mesh_type=mesh_type,
            ground_truth=ground_truth,
            metadata=meta,
            inference_domain="surface",
            reference_geometry=mesh,
            # DrivAerStar surface *is* the geometry: hand the same mesh to the SDF/geometry branch
            # so the wrappers derive stl_coordinates/faces/centers in memory (no STL file needed).
            geometry=mesh,
        )
