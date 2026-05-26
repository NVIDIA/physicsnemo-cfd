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

"""
Standalone distributed inference pipeline for the DoMINO surrogate.

Given a `.mdlus` checkpoint (which carries the model architecture and
weights) plus a co-located `scaling_factors.pkl` (the per-field
non-dimensionalization statistics from training), this module evaluates
the model on a user-supplied STL geometry and returns surface predictions
and per-cell drag sensitivities.

The bounding boxes used for spatial sampling are read from a small Hydra
config (`conf/config.yaml`); everything else lives in the checkpoint
bundle.
"""

import pickle
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import hydra
import numpy as np
import pyvista as pv
import torch
import tyro
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.domino.model import DoMINO
from physicsnemo.models.domino.utils import unnormalize

from design_datapipe import DesignDatapipe
from utilities.download import download
from utilities.mesh_postprocessing import laplacian_smoothing


@dataclass
class _ScalingFactors:
    """Local shim for the upstream DoMINO `ScalingFactors` dataclass.

    The HF checkpoint bundle ships a `scaling_factors.pkl` that is a
    pickled instance of `utils.ScalingFactors` from the upstream
    physicsnemo DoMINO training example. We don't want to depend on the
    training scaffolding here, so we mirror just the attribute shape
    that `pickle` needs to rehydrate the instance.

    Attributes:
        mean: Per-field mean arrays, keyed by e.g. ``"surface_fields"``,
            ``"volume_fields"``, ``"stl_centers"``, etc.
        std: Per-field standard deviations, same keys as ``mean``.
        min_val: Per-field minima, same keys as ``mean``.
        max_val: Per-field maxima, same keys as ``mean``.
        field_keys: List of field keys that statistics were computed for.
    """

    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]
    min_val: dict[str, np.ndarray]
    max_val: dict[str, np.ndarray]
    field_keys: list[str]


class _ScalingFactorsUnpickler(pickle.Unpickler):
    """Unpickler that maps the upstream `utils.ScalingFactors` symbol
    onto our local `_ScalingFactors` shim, so we can read the HF
    `scaling_factors.pkl` without importing the training example.
    """

    def find_class(self, module: str, name: str) -> type:
        if name == "ScalingFactors":
            return _ScalingFactors
        return super().find_class(module, name)


def _load_scaling_factors(path: Path) -> _ScalingFactors:
    """Load a `scaling_factors.pkl` from disk via `_ScalingFactorsUnpickler`."""
    with open(path, "rb") as f:
        return _ScalingFactorsUnpickler(f).load()


def _unwrap_model(model: torch.nn.Module) -> DoMINO:
    """Peel off `DistributedDataParallel` and `torch.compile` wrappers.

    Returns the underlying `DoMINO` instance so we can read
    architecture attributes (e.g. ``grid_resolution``) directly.

    The wrap order applied in :meth:`DoMINOInference.__post_init__` is
    ``DDP(OptimizedModule(DoMINO))`` (DDP outermost, when used at all),
    so we peel in that order.
    """
    inner = getattr(model, "module", model)  # DDP wrapper
    inner = getattr(inner, "_orig_mod", inner)  # torch.compile wrapper
    return inner  # ty: ignore[invalid-return-type]


@dataclass
class DoMINOInference:
    """Distributed inference pipeline for DoMINO on an automotive aero case.

    The model architecture and weights come entirely from the
    `.mdlus` checkpoint (via :meth:`DoMINO.from_checkpoint`). The
    per-field unnormalization factors come from a
    `scaling_factors.pkl` placed next to the checkpoint. `cfg` is only
    consumed for the spatial sampling/cropping bounding boxes
    (`cfg.data.bounding_box`, `cfg.data.bounding_box_surface`).

    Attributes:
        cfg: Hydra configuration providing `data.bounding_box` and
            `data.bounding_box_surface`. Bounding boxes should be
            consistent with the training domain of the checkpoint.
        model_checkpoint_path: Path to a `.mdlus` archive produced by
            `physicsnemo.Module.save`. A `scaling_factors.pkl` is
            expected in the same directory.
        dist: Distributed manager for multi-GPU inference. If `None`,
            runs on a single device.
        device: PyTorch device for computation. Auto-detected if not
            specified.
        model: Pre-built DoMINO instance. Auto-constructed from the
            checkpoint if not provided.

    See Also:
        DesignDatapipe: Data preprocessing pipeline for DoMINO inputs.
        DoMINO: The underlying model architecture.
    """

    cfg: DictConfig
    model_checkpoint_path: Path | str
    dist: DistributedManager | None = None
    device: torch.device | None = None  # If not set, default set in __post_init__
    model: torch.nn.Module | None = None  # If not set, constructed in __post_init__

    def __post_init__(self):
        if self.device is None:  # Sets a default device, if not specified
            if self.dist is not None:
                self.device = self.dist.device
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        if self.model is None:
            ### Construct the model from the checkpoint's own saved
            ### `__init__` args (stored in `args.json` inside the
            ### `.mdlus` zip). This means the local `cfg` doesn't need
            ### to know any model hyperparameters - they all travel with
            ### the checkpoint.
            self.model = DoMINO.from_checkpoint(
                str(self.model_checkpoint_path), strict=True
            )
            self.model = self.model.to(self.device).eval()

            for param in self.model.parameters():
                param.requires_grad = False

            if (self.dist is not None) and (self.dist.world_size > 1):
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[self.dist.local_rank],
                    output_device=self.dist.device,
                    broadcast_buffers=self.dist.broadcast_buffers,
                    find_unused_parameters=self.dist.find_unused_parameters,
                    gradient_as_bucket_view=True,
                    static_graph=True,
                )

    def _bbox(self, key: str) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Read a `(min, max)` bounding-box pair from ``self.cfg.data[key]``.

        Args:
            key: A key under ``cfg.data`` whose value has ``.min`` and
                ``.max`` 3-vectors, e.g. ``"bounding_box"`` or
                ``"bounding_box_surface"``.

        Returns:
            ``(min_xyz, max_xyz)`` as a pair of float32 arrays of shape
            ``(3,)``.

        Notes:
            Bad shapes or missing keys surface as OmegaConf / NumPy errors
            without further wrapping; OmegaConf's own messages are already
            specific.
        """
        spec = self.cfg.data[key]
        return (
            np.array(spec.min, dtype=np.float32),
            np.array(spec.max, dtype=np.float32),
        )

    @cached_property
    def bounding_box_volume_min_max(
        self,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Volume (computational-domain) bounding box from ``cfg.data.bounding_box``."""
        return self._bbox("bounding_box")

    @cached_property
    def bounding_box_surface_min_max(
        self,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Surface bounding box from ``cfg.data.bounding_box_surface``."""
        return self._bbox("bounding_box_surface")

    @cached_property
    def _scaling_factors(self) -> _ScalingFactors:
        """Per-field statistics loaded from `scaling_factors.pkl`.

        The pickle file is expected to live in the same directory as
        the `.mdlus` checkpoint. It is produced by the upstream DoMINO
        training pipeline (see `compute_statistics.py` in the
        physicsnemo DoMINO example).

        Raises:
            FileNotFoundError: If `scaling_factors.pkl` is not found
                next to the checkpoint. The error message points the
                user at the HF bundle layout described in the README.
        """
        path = Path(self.model_checkpoint_path).parent / "scaling_factors.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Expected {path!s} alongside the .mdlus checkpoint, but "
                f"it does not exist. Download the matching scaling factors "
                f"from the same source as your checkpoint (see README)."
            )
        return _load_scaling_factors(path)

    def _build_factors(self, key: str) -> torch.Tensor:
        """Stack `[max, min]` rows for one field key into a `(2, N)` tensor.

        Args:
            key: Field key inside the scaling-factors pickle, e.g.
                ``"surface_fields"`` or ``"volume_fields"``.

        Returns:
            A `(2, num_components)` float32 tensor on `self.device`,
            laid out as ``[max_row, min_row]`` so that
            :func:`physicsnemo.models.domino.utils.unnormalize` (which
            expects this `(max, min)` ordering for min-max scaling)
            returns physical units.

        Raises:
            KeyError: If `key` is not present in the loaded
                `scaling_factors.pkl`.
        """
        sf = self._scaling_factors
        if key not in sf.max_val or key not in sf.min_val:
            raise KeyError(
                f"Scaling factors are missing {key!r}; "
                f"available keys: {sorted(sf.max_val)}"
            )
        stacked = np.stack([sf.max_val[key], sf.min_val[key]], axis=0).astype(
            np.float32
        )
        return torch.from_numpy(stacked).to(self.device)

    @cached_property
    def vol_factors(self) -> torch.Tensor:
        """Volume-field unnormalization factors, shape `(2, num_vol_vars)`."""
        return self._build_factors("volume_fields")

    @cached_property
    def surf_factors(self) -> torch.Tensor:
        """Surface-field unnormalization factors, shape `(2, num_surf_vars)`."""
        return self._build_factors("surface_fields")

    def __call__(
        self,
        mesh: pv.PolyData,
        stream_velocity: float = 38.889,
        stencil_size: int = 7,
        air_density: float = 1.205,
        verbose: bool = False,
    ) -> dict[str, np.ndarray]:
        """Performs DoMINO inference on a given geometry to predict aerodynamic quantities.

        This method takes a PyVista mesh representing a 3D geometry and computes the
        aerodynamic predictions using the DoMINO model. It handles the data preprocessing,
        model inference, and post-processing of results.

        Args:
            mesh: PyVista PolyData mesh representing the 3D geometry to analyze
            stream_velocity: Inlet flow velocity in m/s. Defaults to 38.889 m/s.
            stencil_size: Number of neighboring points to consider for surface calculations.
                Defaults to 7.
            air_density: Air density in kg/m³. Defaults to 1.205 kg/m³.
            verbose: Whether to print verbose output. Defaults to False.

        Returns:
            dict: Dictionary containing the following keys:
                - 'geometry_coordinates': (n_points, 3) point coordinates sampled internally
                - 'geometry_sensitivity': (n_points, 3) d(-drag)/dX vectors per point
                - 'pred_surf_pressure': (n_faces,) predicted surface pressure [Pa]
                - 'pred_surf_wall_shear_stress': (n_faces, 3) predicted wall shear stress [τx, τy, τz] [Pa]
                - 'aerodynamic_force': (3,) integrated aerodynamic force [Fx, Fy, Fz] [N]

        Example:
            >>> import pyvista as pv
            >>> from main import DoMINOInference
            >>>
            >>> mesh = pv.read("car.stl")
            >>>
            >>> domino = DoMINOInference(
            ...     cfg=cfg,
            ...     model_checkpoint_path="./checkpoints/DoMINO.0.501.mdlus",
            ... )
            >>>
            >>> results = domino(
            ...     mesh=mesh,
            ...     stream_velocity=30.0,
            ...     stencil_size=7,
            ...     air_density=1.205,
            ... )
            >>>
            >>> forces = results['aerodynamic_force']
            >>> print(f"Drag force: {forces[0]:.2f} N")
        """
        torch.random.manual_seed(0)

        inner_model = _unwrap_model(self.model)
        datapipe = DesignDatapipe(
            mesh=mesh,
            bounding_box=self.bounding_box_volume_min_max,
            bounding_box_surface=self.bounding_box_surface_min_max,
            grid_resolution=inner_model.grid_resolution,
            stencil_size=stencil_size,
            device=self.device,
            produce_volume_inputs=inner_model.output_features_vol is not None,
        )
        dataloader = torch.utils.data.DataLoader(
            datapipe, batch_size=2**13, shuffle=False
        )

        input_dict: dict[str, torch.Tensor] = {
            k: torch.unsqueeze(v, dim=0) for k, v in datapipe.out_dict.items()
        }

        ### The DoMINO surrogate carries a `global_params_*` block whose
        ### channel order is fixed at training time. For the published
        ### `DoMINO.0.501.mdlus` checkpoint that order is
        ### `[inlet_velocity, air_density]`. With `encode_parameters=False`
        ### the model only uses these as references for non-dimensionalization,
        ### but `DoMINO.forward` still validates that both keys are present.
        ### Final shape is `1 x 2 x 1`: a leading batch axis, then the
        ### two global parameters as a column vector.
        globals_col = torch.tensor(
            [[stream_velocity], [air_density]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        input_dict["global_params_values"] = globals_col
        input_dict["global_params_reference"] = globals_col

        aerodynamic_force = np.zeros(3, dtype=np.float32)
        pred_surf_batches: list[np.ndarray] = []
        geometry_coordinates = (
            input_dict["geometry_coordinates"].detach().cpu().numpy()[0]
        )
        geometry_sensitivity: np.ndarray = np.zeros_like(geometry_coordinates)

        for sample_batched in tqdm(
            dataloader, desc="Processing batches", disable=not verbose
        ):
            ### Splice the surface-side tensors for this batch into the
            ### otherwise-fixed input dict. The exact key set lives on
            ### the datapipe so the two stay in lockstep.
            input_dict_batch: dict[str, torch.Tensor] = {
                **input_dict,
                **{
                    k: torch.unsqueeze(sample_batched[k], dim=0)
                    for k in DesignDatapipe.SURFACE_KEYS
                },
            }
            input_dict_batch["geometry_coordinates"].requires_grad_(True)

            ### `DoMINO.forward` returns `(vol_pred, surf_pred)`; for a
            ### surface-only checkpoint `vol_pred is None`. We discard it
            ### with `_` so we don't retain a reference into the autograd
            ### graph and PyTorch can free the primal values between batches.
            _, prediction_surf_batch = self.model(input_dict_batch)

            prediction_surf_batch = (
                unnormalize(
                    prediction_surf_batch, self.surf_factors[0], self.surf_factors[1]
                )
                * stream_velocity**2.0
                * air_density
            )
            surface_areas_batch = input_dict_batch["surface_areas"][0]
            surface_normals_batch = input_dict_batch["surface_normals"][0]
            pressure_batch = prediction_surf_batch[0][:, 0]
            wall_shear_stress_batch = prediction_surf_batch[0][:, 1:4]

            aerodynamic_force_batch = torch.sum(
                surface_areas_batch[:, None]
                * (
                    surface_normals_batch * pressure_batch[:, None]  # Pressure
                    - wall_shear_stress_batch  # Wall shear stress
                ),
                dim=0,  # Sums over all points in the batch
            )
            drag_force_batch = aerodynamic_force_batch[0]
            (
                -1 * drag_force_batch
            ).backward()  # Vectors represent how you should modify the geometry to *reduce* drag

            # Compute the sensitivity of the drag force to the geometry coordinates, from this batch
            geometry_sensitivity_batch = input_dict_batch["geometry_coordinates"].grad[
                0
            ]

            geometry_sensitivity += geometry_sensitivity_batch.cpu().detach().numpy()
            aerodynamic_force += aerodynamic_force_batch.cpu().detach().numpy()

            pred_surf_batches.append(prediction_surf_batch[0].detach().cpu().numpy())

        pred_surf = np.concatenate(pred_surf_batches, 0)

        return {
            "geometry_coordinates": geometry_coordinates,
            "geometry_sensitivity": geometry_sensitivity,
            "pred_surf_pressure": pred_surf[:, 0],
            "pred_surf_wall_shear_stress": pred_surf[:, 1:4],
            "aerodynamic_force": aerodynamic_force,
        }

    @staticmethod
    def postprocess_point_sensitivities(
        results: dict[str, np.ndarray], mesh: pv.PolyData, n_laplacian_iters: int = 20
    ) -> dict[str, np.ndarray]:
        """Postprocess the raw geometry sensitivities to compute normal and smoothed sensitivities.

        This function takes the raw geometry sensitivities and computes:
        1. Normal sensitivities by projecting onto cell normals
        2. Full sensitivity vectors by scaling cell normals by normal sensitivities
        3. Smoothed versions of both using Laplacian smoothing

        Parameters
        ----------
        results : dict[str, np.ndarray]
            Dictionary containing the raw results from the forward pass, including:
            - geometry_sensitivity: Raw sensitivity vectors for each cell (n_cells, 3)
            - Other keys are preserved in the output
        mesh : pv.PolyData
            PyVista mesh containing the geometry and cell normals
        n_laplacian_iters : int, optional
            Number of Laplacian smoothing iterations to apply, by default 20

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing processed sensitivities:
            - raw_sensitivity_cells: Original geometry sensitivity vectors (n_cells, 3)
            - raw_sensitivity_normal_cells: Scalar sensitivities projected onto cell normals (n_cells,)
            - smooth_sensitivity_point: Laplacian-smoothed vector sensitivity field on points (n_points, 3)
            - smooth_sensitivity_normal_point: Laplacian-smoothed normal-component sensitivities on points (n_points,)
            - smooth_sensitivity_cell: Point-smoothed vector field transferred to cells (n_cells, 3)
            - smooth_sensitivity_normal_cell: Point-smoothed normal-component sensitivities on cells (n_cells,)
        """
        raw_sensitivity_cells = mesh.cell_data["geometry_sensitivity"]
        raw_sensitivity_normal_cells = np.einsum(
            "ij,ij->i",
            raw_sensitivity_cells,
            mesh.cell_normals,
        )

        mesh_pointdata = pv.PolyData(mesh.points, mesh.faces)
        mesh_pointdata.cell_data["raw_sensitivity_normal_cells"] = (
            raw_sensitivity_normal_cells
        )
        mesh_pointdata = mesh_pointdata.cell_data_to_point_data()

        smooth_sensitivity_normal_point = laplacian_smoothing(
            mesh_pointdata,
            mesh_pointdata.point_data["raw_sensitivity_normal_cells"],
            location="points",
            iterations=n_laplacian_iters,
        )
        smooth_sensitivity_point = np.einsum(
            "i,ij->ij",
            smooth_sensitivity_normal_point,
            mesh.point_normals,
        )

        mesh_pointdata.clear_data()
        mesh_pointdata.point_data["smooth_sensitivity_normal_point"] = (
            smooth_sensitivity_normal_point
        )
        mesh_pointdata = mesh_pointdata.point_data_to_cell_data()

        smooth_sensitivity_normal_cell = mesh_pointdata.cell_data[
            "smooth_sensitivity_normal_point"
        ]

        smooth_sensitivity_cell = np.einsum(
            "i,ij->ij",
            smooth_sensitivity_normal_cell,
            mesh.cell_normals,
        )

        return {
            "raw_sensitivity_cells": raw_sensitivity_cells,
            "raw_sensitivity_normal_cells": raw_sensitivity_normal_cells,
            "smooth_sensitivity_point": smooth_sensitivity_point,
            "smooth_sensitivity_normal_point": smooth_sensitivity_normal_point,
            "smooth_sensitivity_cell": smooth_sensitivity_cell,
            "smooth_sensitivity_normal_cell": smooth_sensitivity_normal_cell,
        }


_SCRIPT_DIR = Path(__file__).parent
_DEFAULT_STL_PATH = _SCRIPT_DIR / "geometries" / "drivaer_1.stl"
_DEFAULT_STL_URL = (
    "https://huggingface.co/datasets/neashton/drivaerml/"
    "resolve/main/run_1/drivaer_1.stl"
)


def main(
    model_checkpoint_path: Path = _SCRIPT_DIR / "checkpoints" / "DoMINO.0.501.mdlus",
    input_file: Path = _DEFAULT_STL_PATH,
    stream_velocity: float = 38.889,
    stencil_size: int = 7,
    air_density: float = 1.205,
    verbose: bool = True,
) -> None:
    """Run the DoMINO design-sensitivities inference pipeline end to end.

    Loads the checkpoint, preprocesses the input STL, runs the model
    forward to predict surface pressure and wall shear stress, computes
    drag sensitivities via autograd, smooths them, and writes the
    annotated mesh as a `.vtk` next to the input file.

    Args:
        model_checkpoint_path: Path to a DoMINO `.mdlus` archive. A
            matching `scaling_factors.pkl` is expected in the same
            directory.
        input_file: Path to the input STL geometry. If it does not
            exist and equals the default DrivAerML sample path, it is
            downloaded automatically.
        stream_velocity: Free-stream inlet velocity [m/s].
        stencil_size: Number of nearest neighbors used to build the
            surface stencil.
        air_density: Free-stream air density [kg/m^3].
        verbose: Whether to show a tqdm progress bar over the inference
            batches.
    """
    ### [CUDA Memory Management]
    torch.cuda.set_per_process_memory_fraction(0.9)

    ### [Hydra Config Loading]
    config_path = Path(".") / "conf"
    with hydra.initialize(version_base="1.3", config_path=str(config_path)):
        cfg: DictConfig = hydra.compose(config_name="config")

    ### [Distributed Initialization]
    DistributedManager.initialize()
    dist = DistributedManager()

    if dist.world_size > 1:
        torch.distributed.barrier()  # ty: ignore[possibly-unbound-attribute]

    ### [Model Inference Pipeline Setup]
    domino = DoMINOInference(
        cfg=cfg,
        model_checkpoint_path=model_checkpoint_path,
        dist=dist,
    )

    ### [Input File Download or Validation]
    if not input_file.exists():
        ### Auto-download only when the user is asking for the default
        ### sample geometry; any other missing path is an error so the
        ### user notices the typo instead of silently downloading the
        ### wrong file.
        if input_file.resolve() == _DEFAULT_STL_PATH.resolve():
            download(url=_DEFAULT_STL_URL, filename=input_file)
            if not input_file.exists():
                raise FileNotFoundError(
                    f"Failed to download the default STL file: {input_file}"
                )
        else:
            raise FileNotFoundError(
                f"Input file does not exist: {input_file}. "
                "Please provide a valid STL file path."
            )

    ### [Read Mesh and Run Inference]
    mesh: pv.PolyData = pv.read(input_file)  # ty: ignore[invalid-assignment]
    results: dict[str, np.ndarray] = domino(
        mesh=mesh,
        stream_velocity=stream_velocity,
        stencil_size=stencil_size,
        air_density=air_density,
        verbose=verbose,
    )

    ### [Attach Results to Mesh]
    for key, value in results.items():
        if len(value) == mesh.n_cells:
            mesh.cell_data[key] = value
        elif len(value) == mesh.n_points:
            mesh.point_data[key] = value

    ### [Postprocess Sensitivities]
    sensitivity_results: dict[str, np.ndarray] = domino.postprocess_point_sensitivities(
        results=results, mesh=mesh
    )

    for key, value in sensitivity_results.items():
        mesh[key] = value
    mesh.save(input_file.with_suffix(".vtk"))


if __name__ == "__main__":
    tyro.cli(main)
