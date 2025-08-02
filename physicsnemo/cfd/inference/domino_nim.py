# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
import httpx
import numpy as np
import io
import trimesh


def call_domino_nim(
    stl_path: str | Path,
    inference_api_url: str = "http://localhost:8000/v1/infer",
    data: dict[str, str] = None,
    timeout: float = 120.0,
    verbose: bool = True,
    point_cloud: str | Path = None,
    batch_size: str | int = None,
) -> dict[str, np.ndarray | float]:
    """
    Performs a DoMINO NIM inference on a given STL file.

    Parameters
    ----------
    stl_path : str or Path
        Path to the STL geometry file of the vehicle to be analyzed.
    inference_api_url : str, optional
        URL of the DoMINO NIM inference API endpoint. Defaults to a local endpoint: "http://localhost:8000/v1/infer".
        For reference, these are the simulation-related DoMINO API endpoints:
            - /v1/infer: Runs full DoMINO inference (combined volume and surface predictions)
            - /v1/infer/volume: Runs volume-only DoMINO inference
            - /v1/infer/surface: Runs surface-only DoMINO inference
    data : dict of str, optional
        Dictionary of simulation parameters to pass to the inference API. Default parameters:
            - "stream_velocity": "40" (freestream velocity in m/s)
            - "stencil_size": "1" (controls resolution of the simulation)
            - "point_cloud_size": "500000" (number of points in the output)
        Note that values should be strings.
    timeout : float, optional
        Maximum time in seconds to wait for the API response. Defaults to 120 seconds.
    verbose : bool, optional
        Whether to print status messages during execution. Defaults to True.
    point_cloud : str or Path, optional
        Path to the point cloud file (.npy) to use for inference. If provided, the results
        will be computed on this point cloud instead of a random one.
    batch_size : str or int, optional
        Batch size parameter to pass to the inference API.

    Returns
    -------
    dict of str to np.ndarray or float
        Dictionary containing the inference results with the following keys:
        - "sdf": Signed distance field (numpy.ndarray)
        - "coordinates": 3D coordinates of the volume points (numpy.ndarray)
        - "velocity": Velocity vectors at the volume points (numpy.ndarray)
        - "pressure": Static pressure at the volume points (numpy.ndarray)
        - "turbulent-kinetic-energy": Turbulent kinetic energy at the volume points (numpy.ndarray)
        - "turbulent-viscosity": Eddy viscosity at the volume points (numpy.ndarray)
        - "bounding_box_dims": Dimensions of the simulation domain (numpy.ndarray)
        - "surface_coordinates": Coordinates of the surface points (numpy.ndarray)
        - "pressure_surface": Pressure at the surface points (numpy.ndarray)
        - "wall-shear-stress": Wall shear stress vectors at the surface points (numpy.ndarray)
        - "drag_force": Total integrated drag force on the geometry (float)
        - "lift_force": Total integrated lift force on the geometry (float)
    """
    if data is None:
        data = {
            "stream_velocity": "40",
            "stencil_size": "1",
            "point_cloud_size": "500000",
        }

    # Add batch_size to data if provided
    if batch_size is not None:
        data["batch_size"] = str(batch_size)

    input_file = Path(stl_path)
    output_file = input_file.with_stem(input_file.stem + "_single_solid").with_suffix(
        ".stl"
    )

    m = trimesh.load_mesh(input_file)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))

    m.export(output_file)

    # Open the STL file and send it to the NIM
    if point_cloud is not None:
        point_cloud_path = Path(point_cloud)
        with (
            open(output_file, "rb") as stl_file,
            open(point_cloud_path, "rb") as pc_file,
        ):
            files = {
                "design_stl": (str(output_file), stl_file),
                "point_cloud": (point_cloud_path.name, pc_file),
            }

            if verbose:
                print(
                    f"Sending POST request to DoMINO NIM inference API at {inference_api_url}..."
                )
            try:
                response = httpx.post(
                    inference_api_url,
                    files=files,
                    data=data,
                    timeout=timeout,
                )
            except httpx.ConnectError as e:
                raise RuntimeError(
                    f"Failed to connect to DoMINO NIM inference API at {inference_api_url}. "
                    f"Please ensure the API is running and accessible. Error: {e}"
                )
            except httpx.TimeoutException as e:
                raise RuntimeError(
                    f"Timeout while connecting to DoMINO NIM inference API at {inference_api_url}. "
                    f"Please ensure the API is running and accessible. Error: {e}"
                )
    else:
        with open(output_file, "rb") as stl_file:
            files = {
                "design_stl": (str(output_file), stl_file),
            }

            if verbose:
                print(
                    f"Sending POST request to DoMINO NIM inference API at {inference_api_url}..."
                )
            try:
                response = httpx.post(
                    inference_api_url,
                    files=files,
                    data=data,
                    timeout=timeout,
                )
            except httpx.ConnectError as e:
                raise RuntimeError(
                    f"Failed to connect to DoMINO NIM inference API at {inference_api_url}. "
                    f"Please ensure the API is running and accessible. Error: {e}"
                )
            except httpx.TimeoutException as e:
                raise RuntimeError(
                    f"Timeout while connecting to DoMINO NIM inference API at {inference_api_url}. "
                    f"Please ensure the API is running and accessible. Error: {e}"
                )

    # Check if the request was successful
    if response.status_code != 200:
        raise RuntimeError(
            f"DoMINO NIM inference failed with status code {response.status_code}:\n{response.text}"
        )

    # Load the response content into a NumPy array
    with np.load(io.BytesIO(response.content)) as output_data:
        output_dict = {key: output_data[key] for key in output_data.keys()}

    # Print the keys of the output dictionary
    if verbose:
        print(f"Inference complete. Output keys: {output_dict.keys()}.")

    return output_dict


if __name__ == "__main__":
    output_dict = call_domino_nim(
        stl_path="./drivaer_4.stl",
        inference_api_url="http://localhost:8000/v1/infer",
    )
