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

import pyvista as pv
import numpy as np
import vtk


def compute_streamlines(data, field, *, surface_streamlines: bool = True):
    """Compute streamlines through the mesh.

    Parameters
    ----------
    data :
        PyVista Dataset
    field :
        Field (str) to use while computing the streamlines.
    surface_streamlines :
        Passed to PyVista ``streamlines_from_source``. Use ``True`` for surface/boundary-aligned
        integration (e.g. wall shear on PolyData); ``False`` for full 3-D volume vector fields.

    Returns
    -------
    pyvista.DataSet
        Streamlines from ``streamlines_from_source``.
    """

    # Convert cell data to point data to create streamlines more robustly
    data = data.cell_data_to_point_data(pass_cell_data=True)

    # generate seed points
    poisson_sampler = vtk.vtkPoissonDiskSampler()
    poisson_sampler.SetInputData(data)
    poisson_sampler.SetRadius(0.070)
    poisson_sampler.Update()
    sampled_points = poisson_sampler.GetOutput()
    seed_cloud = pv.wrap(sampled_points)

    streamlines = data.streamlines_from_source(
        vectors=field,  # The name of the vector field
        source=seed_cloud,
        max_steps=1000,
        max_length=10,  # Control how long the streamlines are
        integration_direction="both",
        terminal_speed=1e-12,
        surface_streamlines=surface_streamlines,
    )

    return streamlines
