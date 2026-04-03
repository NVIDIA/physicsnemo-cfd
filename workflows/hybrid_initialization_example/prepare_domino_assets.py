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

from pathlib import Path
from utils import download

example_dir = Path(__file__).parent

# Downloads the vehicle STL file for DrivAerML Run 4
download(
    url="https://huggingface.co/datasets/neashton/drivaerml/resolve/main/run_4/drivaer_4.stl",
    filename=example_dir / "from_domino" / "vehicle.stl",
)

# Performs a DoMINO inference on the vehicle.stl file
from physicsnemo.cfd.evaluation.nims import call_domino_nim

output_dict = call_domino_nim(
    stl_path=example_dir / "from_domino" / "vehicle.stl",
    inference_api_url="http://localhost:8000/v1/infer",
    data={
        "stream_velocity": "38.889",
        "stencil_size": "1",
        "point_cloud_size": "16000000",
    },
)

import pyvista as pv

data = pv.PointSet(output_dict["coordinates"][0]).cast_to_polydata()
data["UMeanTrimPred"] = output_dict["velocity"][0]
data["pMeanTrimPred"] = output_dict["pressure"][0]
data["TKEPred"] = output_dict["turbulent_kinetic_energy"][0]
data["nutMeanTrimPred"] = output_dict["turbulent_viscosity"][0]

data.save(example_dir / "from_domino" / "predicted_flow.vtk")
