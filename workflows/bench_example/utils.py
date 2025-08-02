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

import json
import os
import re
import pyvista as pv
import pandas as pd
import warnings
import numpy as np

from physicsnemo.cfd.bench.metrics.aero_forces import compute_drag_and_lift
from physicsnemo.cfd.bench.metrics.l2_errors import (
    compute_l2_errors,
    compute_area_weighted_l2_errors,
)
from physicsnemo.cfd.bench.metrics.streamlines import compute_streamlines
from physicsnemo.cfd.bench.interpolation.interpolate_mesh_to_pc import (
    interpolate_mesh_to_pc,
)
from physicsnemo.cfd.bench.metrics.physics import (
    compute_continuity_residuals,
    compute_momentum_residuals,
)
from physicsnemo.cfd.bench.visualization.utils import (
    get_visible_point_indices,
    plot_field_comparisons,
)


def load_mapping(s):
    if os.path.isfile(s):
        with open(s, "r") as f:
            return json.load(f)
    return json.loads(s)


def process_surface_results(filenames, field_mapping, compute_projections=False):
    mesh_filename, pc_filename = filenames[0], filenames[1]
    results = {}
    if pc_filename is None:
        print(f"Processing: {mesh_filename}")
    else:
        print(f"Processing: {mesh_filename}, {pc_filename}")
    # Fetch the run number from the filename
    run_idx = re.search(r"(\d+)(?=\D*$)", mesh_filename).group()
    results["run_idx"] = run_idx

    # Read mesh
    reader = pv.get_reader(mesh_filename)
    reader.disable_all_point_arrays()
    reader.disable_all_cell_arrays()

    for field in field_mapping.values():
        if field in reader.point_array_names:
            reader.enable_point_array(field)
        elif field in reader.cell_array_names:
            reader.enable_cell_array(field)

    mesh = reader.read()
    mesh = mesh.point_data_to_cell_data(pass_point_data=True)

    # compute drag and lift coefficients
    (
        results["Cd_true"],
        results["Cd_p_true"],
        results["Cd_f_true"],
        results["Cl_true"],
        results["Cl_p_true"],
        results["Cl_f_true"],
    ) = compute_drag_and_lift(
        mesh,
        pressure_field=field_mapping["p"],
        wss_field=field_mapping["wallShearStress"],
    )

    (
        results["Cd_pred"],
        results["Cd_p_pred"],
        results["Cd_f_pred"],
        results["Cl_pred"],
        results["Cl_p_pred"],
        results["Cl_f_pred"],
    ) = compute_drag_and_lift(
        mesh,
        pressure_field=field_mapping["pPred"],
        wss_field=field_mapping["wallShearStressPred"],
    )

    # compute L2 errors
    results["l2_errors"] = compute_l2_errors(
        mesh,
        [
            field_mapping["p"],
            field_mapping["wallShearStress"],
        ],
        [
            field_mapping["pPred"],
            field_mapping["wallShearStressPred"],
        ],
        dtype="cell",
    )
    results["l2_errors_area_wt"] = compute_area_weighted_l2_errors(
        mesh,
        [
            field_mapping["p"],
            field_mapping["wallShearStress"],
        ],
        [
            field_mapping["pPred"],
            field_mapping["wallShearStressPred"],
        ],
        dtype="cell",
    )

    # compute centerlines
    slice_y_0 = mesh.slice(normal="y", origin=(0, 0, 0))
    results["centerline_top"] = slice_y_0.clip(
        normal="z", origin=(0, 0, 0.4), invert=False
    )
    results["centerline_bottom"] = slice_y_0.clip(
        normal="z", origin=(0, 0, 0.4), invert=True
    )

    if pc_filename is not None:
        # compute pc interpolations and results
        pc = pv.read(pc_filename)

        # interpolate true results from mesh because PCs don't have them
        pc = interpolate_mesh_to_pc(
            pc, mesh, [field_mapping["p"], field_mapping["wallShearStress"]]
        )

        results["l2_errors_pc"] = compute_l2_errors(
            pc,
            [
                field_mapping["p"],
                field_mapping["wallShearStress"],
            ],
            [
                field_mapping["pPred"],
                field_mapping["wallShearStressPred"],
            ],
            dtype="point",
        )

        # compute drag and lift coefficients
        (
            results["Cd_true_pc"],
            results["Cd_p_true_pc"],
            results["Cd_f_true_pc"],
            results["Cl_true_pc"],
            results["Cl_p_true_pc"],
            results["Cl_f_true_pc"],
        ) = compute_drag_and_lift(
            pc,
            pressure_field=field_mapping["p"],
            wss_field=field_mapping["wallShearStress"],
            dtype="point",
        )

        (
            results["Cd_pred_pc"],
            results["Cd_p_pred_pc"],
            results["Cd_f_pred_pc"],
            results["Cl_pred_pc"],
            results["Cl_p_pred_pc"],
            results["Cl_f_pred_pc"],
        ) = compute_drag_and_lift(
            pc,
            pressure_field=field_mapping["pPred"],
            wss_field=field_mapping["wallShearStressPred"],
            dtype="point",
        )
    else:
        results["l2_errors_pc"] = None
        results["Cd_true_pc"] = None
        results["Cd_p_true_pc"] = None
        results["Cd_f_true_pc"] = None
        results["Cl_true_pc"] = None
        results["Cl_p_true_pc"] = None
        results["Cl_f_true_pc"] = None
        results["Cd_pred_pc"] = None
        results["Cd_p_pred_pc"] = None
        results["Cd_f_pred_pc"] = None
        results["Cl_pred_pc"] = None
        results["Cl_p_pred_pc"] = None
        results["Cl_f_pred_pc"] = None

    # Compute the projections
    if compute_projections:
        camera_directions = {
            "XY": "xy",  # +z
            "YZ": "yz",  # +x
            "ZX": "xz",  # +y
            "-XY": (0, 0, -1),  # -z
            "-YZ": (-1, 0, 0),  # -x
            "-ZX": (0, -1, 0),  # -y
        }

        mesh = mesh.cell_data_to_point_data(pass_cell_data=True)
        for cam_dir in camera_directions.keys():
            idx_visible = get_visible_point_indices(mesh, camera_directions[cam_dir])
            if len(idx_visible) == 0:
                continue

            # assemble coordinates
            pts = mesh.points[idx_visible]
            if cam_dir in ["XY", "-XY"]:
                x = pts[:, 0] if cam_dir == "XY" else -pts[:, 0]
                y = pts[:, 1]
            elif cam_dir in ["YZ", "-YZ"]:
                x = pts[:, 1]
                y = pts[:, 2] if cam_dir == "YZ" else -pts[:, 2]
            elif cam_dir in ["ZX", "-ZX"]:
                x = pts[:, 2]
                y = pts[:, 0] if cam_dir == "ZX" else -pts[:, 0]

            # assemble fields
            z = np.zeros_like(x)
            proj_points = np.column_stack((x, y, z))
            proj_pc = pv.PolyData(proj_points)
            proj_pc["p_error"] = np.abs(
                mesh.point_data[field_mapping["pPred"]][idx_visible]
                - mesh.point_data[field_mapping["p"]][idx_visible]
            )
            proj_pc["wallShearStress_error"] = np.linalg.norm(
                np.abs(
                    mesh.point_data[field_mapping["wallShearStressPred"]][idx_visible]
                    - mesh.point_data[field_mapping["wallShearStress"]][idx_visible]
                ),
                axis=1,
            )

            results[f"projection_{cam_dir}"] = proj_pc
    else:
        # Initialize empty projections if not computing them
        for cam_dir in ["XY", "-XY", "YZ", "-YZ", "ZX", "-ZX"]:
            results[f"projection_{cam_dir}"] = None

    return results


def process_volume_results(
    mesh_filename,
    field_mapping,
    bounds=[-3.5, 8.5, -2.25, 2.25, -0.32, 3.00],
    nu=None,
    rho=None,
    compute_continuity_metrics=False,
    compute_momentum_metrics=False,
    compute_on_resampled_grid=False,
    voxel_size=0.03,
):
    results = {}
    print(f"Processing: {mesh_filename}")
    run_idx = re.search(r"(\d+)(?=\D*$)", mesh_filename).group()
    results["run_idx"] = run_idx

    reader = pv.get_reader(mesh_filename)
    reader.disable_all_point_arrays()
    reader.disable_all_cell_arrays()

    for field in field_mapping.values():
        if field in reader.point_array_names:
            reader.enable_point_array(field)
        elif field in reader.cell_array_names:
            reader.enable_cell_array(field)

    mesh = reader.read()
    l2_errors_true_fields = [
        field_mapping["p"],
        field_mapping["U"],
        field_mapping["nut"],
    ]
    l2_errors_pred_fields = [
        field_mapping["pPred"],
        field_mapping["UPred"],
        field_mapping["nutPred"],
    ]

    if compute_continuity_metrics:
        mesh = compute_continuity_residuals(
            mesh,
            true_velocity_field=field_mapping["U"],
            predicted_velocity_field=field_mapping["UPred"],
        )
        l2_errors_true_fields.extend(
            [
                "Continuity",
            ]
        )
        l2_errors_pred_fields.extend(
            [
                "ContinuityPred",
            ]
        )

    if compute_momentum_metrics:
        if nu is None:
            # nu = 1.5881327800829875e-5
            nu = 1.507e-5
            warnings.warn(f"nu is not provided. Defaulting to {nu}")
        if rho is None:
            # rho = 1.225
            rho = 1.0
            warnings.warn(f"rho is not provided. Defaulting to {rho}")

        mesh = compute_momentum_residuals(
            mesh,
            true_velocity_field=field_mapping["U"],
            predicted_velocity_field=field_mapping["UPred"],
            true_pressure_field=field_mapping["p"],
            predicted_pressure_field=field_mapping["pPred"],
            true_nu_field=field_mapping["nut"],
            predicted_nu_field=field_mapping["nutPred"],
            nu=nu,
            rho=rho,
        )
        l2_errors_true_fields.extend(
            [
                "Momentum",
            ]
        )
        l2_errors_pred_fields.extend(["MomentumPred"])

    results["l2_errors"] = compute_l2_errors(
        mesh,
        l2_errors_true_fields,
        l2_errors_pred_fields,
        bounds=bounds,
        dtype="point",
    )

    # compute lines
    y_slice = mesh.slice(normal="y", origin=(0, 0, 0))
    results["centerline_bottom"] = y_slice.slice(normal="z", origin=(0, 0, -0.2376))
    x_slice = mesh.slice(normal="x", origin=(0.35, 0, 0))
    results["front_wheel_wake"] = x_slice.slice(normal="z", origin=(0, 0, -0.2376))
    x_slice = mesh.slice(normal="x", origin=(3.15, 0, 0))
    results["rear_wheel_wake"] = x_slice.slice(normal="z", origin=(0, 0, -0.2376))
    x_slice = mesh.slice(normal="x", origin=(4, 0, 0))
    results["wake_x_4"] = x_slice.slice(normal="y", origin=(0, 0, 0))
    x_slice = mesh.slice(normal="x", origin=(5, 0, 0))
    results["wake_x_5"] = x_slice.slice(normal="y", origin=(0, 0, 0))

    # Resample the volume
    if compute_on_resampled_grid:
        x = np.arange(bounds[0], bounds[1] + voxel_size, voxel_size)
        y = np.arange(bounds[2], bounds[3] + voxel_size, voxel_size)
        z = np.arange(bounds[4], bounds[5] + voxel_size, voxel_size)
        x, y, z = np.meshgrid(x, y, z, indexing="ij")
        grid = pv.StructuredGrid(x, y, z)
        grid = interpolate_mesh_to_pc(
            grid, mesh, field_mapping.values(), mesh_dtype="point"
        )
        results["resampled_volume"] = grid
    else:
        results["resampled_volume"] = None

    return results


def plot_surface_results(filename, field_mapping, output_dir):
    run_idx = re.search(r"(\d+)(?=\D*$)", filename).group()
    reader = pv.get_reader(filename)
    reader.disable_all_point_arrays()
    reader.disable_all_cell_arrays()

    for field in field_mapping.values():
        if field in reader.point_array_names:
            reader.enable_point_array(field)
        elif field in reader.cell_array_names:
            reader.enable_cell_array(field)

    mesh = reader.read()
    mesh = mesh.point_data_to_cell_data()

    default_scalar_bar_args = {
        "title_font_size": 42,
        "label_font_size": 42,
        "width": 0.8,
        "n_labels": 3,
        "position_x": 0.1,
    }

    plotter = plot_field_comparisons(
        mesh,
        true_fields=[
            field_mapping["p"],
            field_mapping["wallShearStress"],
        ],
        pred_fields=[
            field_mapping["pPred"],
            field_mapping["wallShearStressPred"],
        ],
        view="xy",
        plot_vector_components=True,
        scalar_bar_args=default_scalar_bar_args,
        cmap="jet",
        lut=20,
    )
    plotter.screenshot(f"./{output_dir}/compare_surface_xy_{run_idx}.png")

    plotter = plot_field_comparisons(
        mesh,
        true_fields=[
            field_mapping["p"],
            field_mapping["wallShearStress"],
        ],
        pred_fields=[
            field_mapping["pPred"],
            field_mapping["wallShearStressPred"],
        ],
        view="yz",
        plot_vector_components=True,
        scalar_bar_args=default_scalar_bar_args,
        cmap="jet",
        lut=20,
    )
    plotter.screenshot(f"./{output_dir}/compare_surface_yz_{run_idx}.png")

    plotter = plot_field_comparisons(
        mesh,
        true_fields=[
            field_mapping["p"],
            field_mapping["wallShearStress"],
        ],
        pred_fields=[
            field_mapping["pPred"],
            field_mapping["wallShearStressPred"],
        ],
        view="xz",
        plot_vector_components=True,
        scalar_bar_args=default_scalar_bar_args,
        cmap="jet",
        lut=20,
    )
    plotter.screenshot(f"./{output_dir}/compare_surface_xz_{run_idx}.png")


def plot_volume_results(
    filename,
    field_mapping,
    output_dir,
    nu=None,
    rho=None,
    compute_continuity_metrics=False,
    compute_momentum_metrics=False,
    bounds=[-3.5, 8.5, -2.25, 2.25, -0.32, 3.00],
):
    run_idx = re.search(r"(\d+)(?=\D*$)", filename).group()
    reader = pv.get_reader(filename)
    reader.disable_all_point_arrays()
    reader.disable_all_cell_arrays()

    for field in field_mapping.values():
        if field in reader.point_array_names:
            reader.enable_point_array(field)
        elif field in reader.cell_array_names:
            reader.enable_cell_array(field)

    mesh = reader.read()

    default_scalar_bar_args = {
        "title_font_size": 42,
        "label_font_size": 42,
        "width": 0.8,
        "n_labels": 3,
        "position_x": 0.1,
    }

    if compute_continuity_metrics:
        mesh = compute_continuity_residuals(
            mesh,
            true_velocity_field=field_mapping["U"],
            predicted_velocity_field=field_mapping["UPred"],
        )

    if compute_momentum_metrics:
        if nu is None:
            # nu = 1.5881327800829875e-5
            nu = 1.507e-5
            warnings.warn(f"nu is not provided. Defaulting to {nu}")
        if rho is None:
            # rho = 1.225
            rho = 1.0
            warnings.warn(f"rho is not provided. Defaulting to {rho}")

        mesh = compute_momentum_residuals(
            mesh,
            true_velocity_field=field_mapping["U"],
            predicted_velocity_field=field_mapping["UPred"],
            true_pressure_field=field_mapping["p"],
            predicted_pressure_field=field_mapping["pPred"],
            true_nu_field=field_mapping["nut"],
            predicted_nu_field=field_mapping["nutPred"],
            nu=nu,
            rho=rho,
        )

    y_slice = mesh.slice(normal="y", origin=(0, 0, 0))
    y_slice = y_slice.clip_box(bounds, invert=False)
    true_fields = [
        field_mapping["p"],
        field_mapping["U"],
        field_mapping["nut"],
    ]
    pred_fields = [
        field_mapping["pPred"],
        field_mapping["UPred"],
        field_mapping["nutPred"],
    ]

    if compute_continuity_metrics:
        true_fields.extend(
            [
                "Continuity",
            ]
        )
        pred_fields.extend(
            [
                "ContinuityPred",
            ]
        )

    if compute_momentum_metrics:
        true_fields.extend(
            [
                "Momentum",
            ]
        )
        pred_fields.extend(
            [
                "MomentumPred",
            ]
        )

    plotter = plot_field_comparisons(
        y_slice,
        true_fields=true_fields,
        pred_fields=pred_fields,
        plot_vector_components=True,
        view="xz",
        dtype="point",
        scalar_bar_args=default_scalar_bar_args,
        cmap="jet",
        lut=20,
    )

    plotter.screenshot(f"./{output_dir}/compare_volume_y_slice_{run_idx}.png")

    z_slice = mesh.slice(normal="z", origin=(0, 0, -0.2376))
    z_slice = z_slice.clip_box(bounds, invert=False)

    plotter = plot_field_comparisons(
        z_slice,
        true_fields=true_fields,
        pred_fields=pred_fields,
        plot_vector_components=True,
        view="xy",
        dtype="point",
        scalar_bar_args=default_scalar_bar_args,
        cmap="jet",
        lut=20,
    )

    plotter.screenshot(f"./{output_dir}/compare_volume_z_slice_{run_idx}.png")


def save_results_to_csv(results, filename, columns):
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(filename, index=False)


def load_results_from_csv(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename).to_dict(orient="list")
    return None


def save_polydata(meshes, directory, prefix, run_idx, extension="vtp"):
    for idx, mesh in zip(run_idx, meshes):
        mesh.save(os.path.join(directory, f"{prefix}_{idx}.{extension}"))


def load_polydata(directory, prefix, run_idx, extension="vtp"):
    meshes = []
    for idx in run_idx:
        filepath = os.path.join(directory, f"{prefix}_{idx}.{extension}")
        if os.path.exists(filepath):
            meshes.append(pv.read(filepath))
        else:
            return None
    return meshes
