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

import sys
import os
import pyvista as pv
import glob
import matplotlib.pyplot as plt
import re
from multiprocessing import Pool
import numpy as np
import pandas as pd
import argparse
import json
from functools import partial
import vtk
import numpy as np

vtk.vtkObject.GlobalWarningDisplayOff()


from physicsnemo.cfd.bench.visualization.utils import plot_line, plot_fields
from utils import (
    load_mapping,
    process_volume_results,
    plot_volume_results,
    save_results_to_csv,
    load_results_from_csv,
    save_polydata,
    load_polydata,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the validation results for volume meshes"
    )
    parser.add_argument(
        "sim_results_dir", type=str, help="directory with volume vtu files"
    )
    parser.add_argument(
        "-n",
        "--num-procs",
        type=int,
        default=1,
        help="number of parallel processes to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="volume_benchmarking",
        help="directory to store results",
    )
    parser.add_argument(
        "--contour-plot-ids",
        nargs="+",
        type=str,
        help="run indices to plot contour plots for",
    )
    parser.add_argument(
        "--field-mapping",
        type=load_mapping,
        default={
            "p": "pMeanTrim",
            "U": "UMeanTrim",
            "nut": "nutMeanTrim",
            "pPred": "pMeanTrimPred",
            "UPred": "UMeanTrimPred",
            "nutPred": "nutMeanTrimPred",
        },
        help='mapping of field names to use for benchmarking, either as a path to a json file or a json string. Example: --field-mapping \'{"p": "pMeanTrim", "wallShearStress": "wallShearStressMeanTrim", "pPred": "pMeanTrimPred", "wallShearStressPred": "wallShearStressMeanTrimPred"}\'',
    )
    parser.add_argument(
        "--plot-aggregate-volume-errors",
        action="store_true",
        default=False,
        help="whether to plot the aggregated volume results",
    )
    parser.add_argument(
        "--compute-continuity-metrics",
        action="store_true",
        default=False,
        help="whether to compute the continuity metrics",
    )
    parser.add_argument(
        "--compute-momentum-metrics",
        action="store_true",
        default=False,
        help="whether to compute the momentum metrics",
    )

    args = parser.parse_args()

    compute_continuity_metrics = args.compute_continuity_metrics
    compute_momentum_metrics = args.compute_momentum_metrics
    compute_on_resampled_grid = args.plot_aggregate_volume_errors
    plot_continuity_metrics = args.compute_continuity_metrics
    plot_momentum_metrics = args.compute_momentum_metrics
    sim_mesh_results_dir = args.sim_results_dir

    mesh_filenames = glob.glob(os.path.join(sim_mesh_results_dir, "*.vtu"))
    run_idx_list = []
    for filename in mesh_filenames:
        run_idx_match = re.search(r"(\d+)(?=\D*$)", filename)
        if run_idx_match:
            run_idx = run_idx_match.group()
            run_idx_list.append(run_idx)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Check if results already exist
    l2_errors_csv = os.path.join(output_dir, "l2_errors.csv")
    l2_errors = load_results_from_csv(l2_errors_csv)

    centerlines_bottom = load_polydata(output_dir, "centerline_bottom", run_idx_list)
    front_wheel_wakes = load_polydata(output_dir, "front_wheel_wake", run_idx_list)
    rear_wheel_wakes = load_polydata(output_dir, "rear_wheel_wake", run_idx_list)
    wakes_x_4 = load_polydata(output_dir, "wake_x_4", run_idx_list)
    wakes_x_5 = load_polydata(output_dir, "wake_x_5", run_idx_list)

    if args.plot_aggregate_volume_errors:
        resampled_volumes = load_polydata(
            output_dir, "resampled_volume", run_idx_list, extension="vtk"
        )

    if (
        not l2_errors
        or not centerlines_bottom
        or not front_wheel_wakes
        or not rear_wheel_wakes
        or not wakes_x_4
        or not wakes_x_5
    ):
        # Process the data if any of the required files are missing
        with Pool(processes=args.num_procs) as pool:
            mesh_results = pool.map(
                partial(
                    process_volume_results,
                    field_mapping=args.field_mapping,
                    compute_continuity_metrics=compute_continuity_metrics,
                    compute_momentum_metrics=compute_momentum_metrics,
                    compute_on_resampled_grid=compute_on_resampled_grid,
                ),
                mesh_filenames,
            )

        # Prepare data for saving
        l2_errors = {"run_idx": []}
        for key in mesh_results[0]["l2_errors"].keys():
            l2_errors[key] = []

        centerlines_bottom = []
        front_wheel_wakes = []
        rear_wheel_wakes = []
        wakes_x_4 = []
        wakes_x_5 = []
        resampled_volumes = []

        for mesh_result in mesh_results:
            l2_errors["run_idx"].append(mesh_result["run_idx"])
            for key, value in mesh_result["l2_errors"].items():
                l2_errors[key].append(value)

            centerlines_bottom.append(mesh_result["centerline_bottom"])
            front_wheel_wakes.append(mesh_result["front_wheel_wake"])
            rear_wheel_wakes.append(mesh_result["rear_wheel_wake"])
            wakes_x_4.append(mesh_result["wake_x_4"])
            wakes_x_5.append(mesh_result["wake_x_5"])
            resampled_volumes.append(mesh_result["resampled_volume"])

        # Save results to CSV
        save_results_to_csv(l2_errors, l2_errors_csv, l2_errors.keys())

        # Save vtps
        save_polydata(
            centerlines_bottom, output_dir, "centerline_bottom", l2_errors["run_idx"]
        )
        save_polydata(
            front_wheel_wakes, output_dir, "front_wheel_wake", l2_errors["run_idx"]
        )
        save_polydata(
            rear_wheel_wakes, output_dir, "rear_wheel_wake", l2_errors["run_idx"]
        )
        save_polydata(wakes_x_4, output_dir, "wake_x_4", l2_errors["run_idx"])
        save_polydata(wakes_x_5, output_dir, "wake_x_5", l2_errors["run_idx"])

        # Save resampled volumes
        if args.plot_aggregate_volume_errors:
            save_polydata(
                resampled_volumes,
                output_dir,
                "resampled_volume",
                run_idx_list,
                extension="vtk",
            )

    else:
        # Load results from saved CSVs
        mesh_results = []
        for i in range(len(l2_errors["run_idx"])):
            mesh_result = {
                "run_idx": l2_errors["run_idx"][i],
                "l2_errors": {
                    key: l2_errors[key][i] for key in l2_errors if key != "run_idx"
                },
                "centerline_bottom": centerlines_bottom[i],
                "front_wheel_wake": front_wheel_wakes[i],
                "rear_wheel_wake": rear_wheel_wakes[i],
                "wake_x_4": wakes_x_4[i],
                "wake_x_5": wakes_x_5[i],
            }

            if args.plot_aggregate_volume_errors:
                mesh_result.update(
                    {
                        "resampled_volume": resampled_volumes[i],
                    }
                )

            mesh_results.append(mesh_result)

    centerlines_bottom = []
    front_wheel_wakes = []
    rear_wheel_wakes = []
    wakes_x_4 = []
    wakes_x_5 = []
    mean_l2_errors = {}

    for key in mesh_results[0]["l2_errors"].keys():
        mean_l2_errors[key] = []

    for mesh_result in mesh_results:
        centerlines_bottom.append(mesh_result["centerline_bottom"])
        front_wheel_wakes.append(mesh_result["front_wheel_wake"])
        rear_wheel_wakes.append(mesh_result["rear_wheel_wake"])
        wakes_x_4.append(mesh_result["wake_x_4"])
        wakes_x_5.append(mesh_result["wake_x_5"])

        for key, value in mesh_result["l2_errors"].items():
            mean_l2_errors[key].append(value)

    for key, values in mean_l2_errors.items():
        mean_l2_errors[key] = np.mean(np.array(values))

    for key, value in mean_l2_errors.items():
        print(f"L2 Errors for {key}: {value}")

    fig = plot_line(
        centerlines_bottom,
        plot_coord="x",
        field_true=args.field_mapping["U"],
        field_pred=args.field_mapping["UPred"],
        normalize_factor=38.889,
        coord_trim=(-1.0, 6.0),
        field_trim=(0, 2.0),
        flip=False,
        true_line_kwargs={"color": "red", "alpha": 1 / len(centerlines_bottom)},
        pred_line_kwargs={"color": "green", "alpha": 1 / len(centerlines_bottom)},
        mean_true_line_kwargs={"color": "red", "label": "Mean True"},
        mean_pred_line_kwargs={"color": "green", "label": "Mean Pred"},
        xlabel="X Coordinate",
        ylabel="U / U_ref",
    )
    fig.savefig(f"./{output_dir}/volume_centerline.png")

    fig = plot_line(
        front_wheel_wakes,
        plot_coord="y",
        field_true=args.field_mapping["U"],
        field_pred=args.field_mapping["UPred"],
        normalize_factor=38.889,
        coord_trim=(-1.0, 1.0),
        field_trim=(0, 2.0),
        flip=False,
        true_line_kwargs={"color": "red", "alpha": 1 / len(front_wheel_wakes)},
        pred_line_kwargs={"color": "green", "alpha": 1 / len(front_wheel_wakes)},
        mean_true_line_kwargs={"color": "red", "label": "Mean True"},
        mean_pred_line_kwargs={"color": "green", "label": "Mean Pred"},
        xlabel="Y Coordinate",
        ylabel="U / U_ref",
    )
    fig.savefig(f"./{output_dir}/volume_front_wheel_wake.png")

    fig = plot_line(
        rear_wheel_wakes,
        plot_coord="y",
        field_true=args.field_mapping["U"],
        field_pred=args.field_mapping["UPred"],
        normalize_factor=38.889,
        coord_trim=(-1.0, 1.0),
        field_trim=(0, 2.0),
        flip=False,
        true_line_kwargs={"color": "red", "alpha": 1 / len(rear_wheel_wakes)},
        pred_line_kwargs={"color": "green", "alpha": 1 / len(rear_wheel_wakes)},
        mean_true_line_kwargs={"color": "red", "label": "Mean True"},
        mean_pred_line_kwargs={"color": "green", "label": "Mean Pred"},
        xlabel="Y Coordinate",
        ylabel="U / U_ref",
    )
    fig.savefig(f"./{output_dir}/volume_rear_wheel_wake.png")

    fig = plot_line(
        wakes_x_4,
        plot_coord="z",
        field_true=args.field_mapping["U"],
        field_pred=args.field_mapping["UPred"],
        normalize_factor=38.889,
        coord_trim=(-0.5, 1.5),
        field_trim=(0, 2.0),
        flip=True,
        true_line_kwargs={"color": "red", "alpha": 1 / len(wakes_x_4)},
        pred_line_kwargs={"color": "green", "alpha": 1 / len(wakes_x_4)},
        mean_true_line_kwargs={"color": "red", "label": "Mean True"},
        mean_pred_line_kwargs={"color": "green", "label": "Mean Pred"},
        xlabel="Z Coordinate",
        ylabel="U / U_ref",
    )
    fig.savefig(f"./{output_dir}/volume_x_4_wake.png")

    fig = plot_line(
        wakes_x_5,
        plot_coord="z",
        field_true=args.field_mapping["U"],
        field_pred=args.field_mapping["UPred"],
        normalize_factor=38.889,
        coord_trim=(-0.5, 1.5),
        field_trim=(0, 2.0),
        flip=True,
        true_line_kwargs={"color": "red", "alpha": 1 / len(wakes_x_5)},
        pred_line_kwargs={"color": "green", "alpha": 1 / len(wakes_x_5)},
        mean_true_line_kwargs={"color": "red", "label": "Mean True"},
        mean_pred_line_kwargs={"color": "green", "label": "Mean Pred"},
        xlabel="Z Coordinate",
        ylabel="U / U_ref",
    )
    fig.savefig(f"./{output_dir}/volume_x_5_wake.png")

    if args.plot_aggregate_volume_errors:
        true_fields = [
            args.field_mapping["p"],
            args.field_mapping["U"],
            args.field_mapping["nut"],
        ]
        pred_fields = [
            args.field_mapping["pPred"],
            args.field_mapping["UPred"],
            args.field_mapping["nutPred"],
        ]
        error_arrays = {f"{k}_error": [] for k in true_fields}
        for resampled_volume in resampled_volumes:
            for true_field, pred_field in zip(true_fields, pred_fields):
                error = np.abs(
                    resampled_volume.point_data[true_field]
                    - resampled_volume.point_data[pred_field]
                )
                error_arrays[f"{true_field}_error"].append(error)

        resampled_volume = resampled_volumes[0]
        fields_to_plot = []
        for k, v in error_arrays.items():
            mean = np.mean(np.stack(v, axis=0), axis=0)
            std = np.std(np.stack(v, axis=0), axis=0)
            resampled_volume.point_data[f"{k}_mean"] = mean
            resampled_volume.point_data[f"{k}_std"] = std
            fields_to_plot.append(f"{k}_mean")
            fields_to_plot.append(f"{k}_std")

        resampled_volume.save(f"./{output_dir}/aggregate_resampled_volume.vtk")

        y_slice = resampled_volume.slice(normal="y", origin=(0, 0, 0))
        plotter = plot_fields(
            y_slice,
            fields_to_plot,
            plot_vector_components=True,
            view="xz",
            dtype="point",
            cmap="jet",
            lut=20,
            window_size=[1280, 3840],
        )
        plotter.screenshot(f"./{output_dir}/aggregate_volume_y_slice.png")

        z_slice = resampled_volume.slice(normal="z", origin=(0, 0, -0.2376))
        plotter = plot_fields(
            z_slice,
            fields_to_plot,
            plot_vector_components=True,
            view="xy",
            dtype="point",
            cmap="jet",
            lut=20,
            window_size=[1280, 3840],
        )
        plotter.screenshot(f"./{output_dir}/aggregate_volume_z_slice.png")

    if args.contour_plot_ids is not None:
        plot_filenames = []
        for filename in mesh_filenames:
            run_idx = re.search(r"(\d+)(?=\D*$)", filename).group()

            if run_idx in args.contour_plot_ids:
                plot_filenames.append(filename)

        print(f"Plotting contour plots for {args.contour_plot_ids}")
        with Pool(processes=args.num_procs) as pool:
            _ = pool.map(
                partial(
                    plot_volume_results,
                    field_mapping=args.field_mapping,
                    output_dir=args.output_dir,
                    compute_continuity_metrics=plot_continuity_metrics,
                    compute_momentum_metrics=plot_momentum_metrics,
                ),
                plot_filenames,
            )
