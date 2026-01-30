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

import physicsnemo.cfd.hybrid_initialization_tools as hit
from pathlib import Path
import numpy as np
import pyvista as pv

try:
    pwd = Path(__file__).parent
except NameError:
    pwd = Path(".").absolute()  # Fallback for interactive use

# Read the OpenFOAM mesh, which contains the potential flow initialization
print("Loading potential flow flowfield...")
flowfield_pf = hit.Flowfield(
    mesh=hit.openfoam_utils.read_openfoam_mesh(pwd),
    velocity_fieldname="U",
    pressure_fieldname="p",
    k_fieldname="k",
    omega_fieldname="omega",
)

# Read the ML mesh, which contains the DoMINO prediction
print("Loading ML flowfield...")
flowfield_ml = hit.Flowfield(
    mesh=pv.read(pwd / "from_domino" / "predicted_flow.vtk"),
    velocity_fieldname="UMeanTrimPred",
    pressure_fieldname="pMeanTrimPred",
    k_fieldname="TKEPred",
    omega_fieldname="omegaMeanTrimPred",
)

print("In ML flowfield, casting to point data...")
flowfield_ml.mesh = flowfield_ml.mesh.cast_to_pointset()

print("In ML flowfield, trimming out points outside of DoMINO's prediction domain...")
bad_points = (flowfield_ml.mesh[flowfield_ml.pressure_fieldname] == 0) & np.all(
    flowfield_ml.mesh[flowfield_ml.velocity_fieldname] == 0, axis=1
)
print(
    f"Eliminating {np.sum(bad_points)} points outside of DoMINO's prediction domain..."
)
flowfield_ml.mesh = hit.pyvista_utils.mask_pointset(
    flowfield_ml.mesh, np.logical_not(bad_points)
)

print("In ML flowfield, converting k-nut to k-omega...")
flowfield_ml.mesh.point_data[flowfield_ml.omega_fieldname] = (
    np.maximum(flowfield_ml.mesh.point_data[flowfield_ml.k_fieldname], 0.24)
    * 1  # OpenFOAM-assumed density value
    / np.maximum(flowfield_ml.mesh.point_data["nutMeanTrimPred"], 1.507e-5)
)

hybrid_initialization = hit.create_hybrid_initialization(
    flowfield_a=flowfield_ml,
    flowfield_b=flowfield_pf,
    use_topology_from_mesh="b",
    flowfield_a_data_location="point",
    flowfield_b_data_location="cell",
)

print("Writing initialization for OpenFOAM...")
for field in ["U", "p", "k", "omega"]:
    hit.openfoam_utils.replace_internal_field(
        pwd / f"0_org/{field}",
        pwd / f"0_hybrid/{field}",
        hybrid_initialization.mesh.cell_data[field],
    )
    print(f"{field} written.")

visualize = True

if visualize:
    plotter = pv.Plotter(off_screen=True)

    # Create a slice of the mesh at y=0
    slice_y = hybrid_initialization.mesh.slice(normal=[0, 1, 0], origin=[0, 0, 0])

    # Add the slice to the plotter with a nice colormap
    sargs = dict(
        title="Velocity Magnitude (m/s)",
        position_x=0.15,
        position_y=0.05,
        width=0.7,
        height=0.05,
        fmt="%.1f",
        color="black",
    )

    plotter.add_mesh(
        slice_y,
        scalars="U",
        cmap="turbo",
        show_edges=False,
        scalar_bar_args=sargs,
    )

    # Set camera position for orthogonal view focused on the car region
    plotter.camera_position = "xz"  # Top view
    plotter.camera.SetParallelProjection(True)  # Orthogonal projection
    plotter.camera.SetFocalPoint([2.0, 0.0, 1.34])  # Look at center of ROI
    plotter.camera.SetViewUp([0.0, 0.0, 1.0])  # Set z as up direction
    plotter.camera.Zoom(12)  # Adjust zoom factor as needed
    plotter.add_title("Hybrid Initialization: Velocity Field (y=0 slice)", font_size=18)
    screenshot_path = pwd / "hybrid_velocity_slice.png"
    plotter.screenshot(str(screenshot_path), return_img=False, window_size=[1920, 1080])
    print(f"Screenshot saved to {screenshot_path}")

    # Close the plotter to free resources
    plotter.close()
