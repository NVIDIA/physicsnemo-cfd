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

"""Model wrapper implementations; registration happens on import."""

from physicsnemo.cfd.evaluation.models.model_registry import register_model
from physicsnemo.cfd.evaluation.models.wrappers.domino import DominoWrapper
from physicsnemo.cfd.evaluation.models.wrappers.fignet import FIGNetWrapper
from physicsnemo.cfd.evaluation.models.wrappers.geotransolver import (
    GeoTransolverWrapper,
)
from physicsnemo.cfd.evaluation.models.wrappers.geotransolver_drivaerstar import (
    GeoTransolverDrivAerStarWrapper,
)
from physicsnemo.cfd.evaluation.models.wrappers.geotransolver_gp import (
    GeoTransolverGPDrivAerStarWrapper,
)
from physicsnemo.cfd.evaluation.models.wrappers.ensemble_drivaerstar import (
    EnsembleDrivAerStarWrapper,
)
from physicsnemo.cfd.evaluation.models.wrappers.mc_perturbation import (
    MCPerturbationDrivAerStarWrapper,
)
from physicsnemo.cfd.evaluation.models.wrappers.surface_baseline import (
    SurfaceBaselineWrapper,
)
from physicsnemo.cfd.evaluation.models.wrappers.transolver import TransolverWrapper
from physicsnemo.cfd.evaluation.models.wrappers.volume_baseline import (
    VolumeBaselineWrapper,
)
from physicsnemo.cfd.evaluation.models.wrappers.xmgn import XMGNWrapper

register_model("fignet_surface", FIGNetWrapper)
register_model("xmgn_surface", XMGNWrapper)
register_model("geotransolver_surface", GeoTransolverWrapper)
register_model("geotransolver_volume", GeoTransolverWrapper)
register_model("geotransolver_drivaerstar_surface", GeoTransolverDrivAerStarWrapper)
register_model("geotransolver_gp_surface", GeoTransolverGPDrivAerStarWrapper)
register_model("mc_perturbation_surface", MCPerturbationDrivAerStarWrapper)
register_model("ensemble_surface", EnsembleDrivAerStarWrapper)
register_model("transolver_surface", TransolverWrapper)
register_model("transolver_volume", TransolverWrapper)
register_model("domino_surface", DominoWrapper)
register_model("domino_volume", DominoWrapper)
register_model("surface_baseline", SurfaceBaselineWrapper)
register_model("volume_baseline", VolumeBaselineWrapper)

__all__ = [
    "FIGNetWrapper",
    "XMGNWrapper",
    "GeoTransolverWrapper",
    "GeoTransolverDrivAerStarWrapper",
    "GeoTransolverGPDrivAerStarWrapper",
    "EnsembleDrivAerStarWrapper",
    "MCPerturbationDrivAerStarWrapper",
    "TransolverWrapper",
    "DominoWrapper",
    "SurfaceBaselineWrapper",
    "VolumeBaselineWrapper",
]
