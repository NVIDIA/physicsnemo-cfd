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

from physicsnemo.cfd.bench.visualization.utils import (
    plot_design_scatter,
    plot_design_trend,
    plot_line,
    plot_projections_hexbin,
)

from utils import (
    load_mapping,
    process_surface_results,
    plot_surface_results,
    save_results_to_csv,
    load_results_from_csv,
    save_polydata,
    load_polydata,
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compute the validation results for surface meshes"
    )
    parser.add_argument(
        "sim_results_dir", type=str, help="directory with surface vtp files"
    )
    parser.add_argument(
        "--pc-results-dir",
        type=str,
        help="directory with point cloud vtp files (optional)",
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
        default="surface_benchmarking",
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
            "wallShearStress": "wallShearStressMeanTrim",
            "pPred": "pMeanTrimPred",
            "wallShearStressPred": "wallShearStressMeanTrimPred",
        },
        help='mapping of field names to use for benchmarking, either as a path to a json file or a json string. Example: --field-mapping \'{"p": "pMeanTrim", "wallShearStress": "wallShearStressMeanTrim", "pPred": "pMeanTrimPred", "wallShearStressPred": "wallShearStressMeanTrimPred"}\'',
    )
    parser.add_argument(
        "--plot-aggregate-surface-errors",
        action="store_true",
        default=False,
        help="whether to plot the aggregated surface results",
    )
    args = parser.parse_args()

    sim_mesh_results_dir = args.sim_results_dir
    pc_results_dir = args.pc_results_dir

    mesh_filenames = glob.glob(os.path.join(sim_mesh_results_dir, "*.vtp"))
    pc_filenames = []
    pc_file_map = {}
    run_idx_list = []

    if args.pc_results_dir:
        for pc_filename in os.listdir(args.pc_results_dir):
            run_idx_match = re.search(r"(\d+)(?=\D*$)", pc_filename)
            if run_idx_match:
                run_idx = run_idx_match.group()
                pc_file_map[run_idx] = os.path.join(args.pc_results_dir, pc_filename)

    # Match mesh filenames to pc filenames
    for filename in mesh_filenames:
        run_idx = re.search(r"(\d+)(?=\D*$)", filename).group()
        run_idx_list.append(run_idx)
        if run_idx in pc_file_map:
            pc_filenames.append(pc_file_map[run_idx])
        else:
            pc_filenames.append(None)

    filenames = list(zip(mesh_filenames, pc_filenames))

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Check if results already exist
    results_csv = os.path.join(output_dir, "results.csv")
    results_pc_csv = os.path.join(output_dir, "results_pc.csv")
    l2_errors_csv = os.path.join(output_dir, "l2_errors.csv")
    l2_errors_area_wt_csv = os.path.join(output_dir, "l2_errors_area_wt.csv")
    l2_errors_pc_csv = os.path.join(output_dir, "l2_errors_pc.csv")

    results = load_results_from_csv(results_csv)
    results_pc = load_results_from_csv(results_pc_csv)
    l2_errors = load_results_from_csv(l2_errors_csv)
    l2_errors_area_wt = load_results_from_csv(l2_errors_area_wt_csv)
    l2_errors_pc = load_results_from_csv(l2_errors_pc_csv)
    top_centerlines = load_polydata(output_dir, "top_centerline", run_idx_list)
    bottom_centerlines = load_polydata(output_dir, "bottom_centerline", run_idx_list)

    if args.plot_aggregate_surface_errors:
        pos_xy_projections = load_polydata(output_dir, "projection_XY", run_idx_list)
        neg_xy_projections = load_polydata(output_dir, "projection_-XY", run_idx_list)
        pos_yz_projections = load_polydata(output_dir, "projection_YZ", run_idx_list)
        neg_yz_projections = load_polydata(output_dir, "projection_-YZ", run_idx_list)
        pos_zx_projections = load_polydata(output_dir, "projection_ZX", run_idx_list)
        neg_zx_projections = load_polydata(output_dir, "projection_-ZX", run_idx_list)

    if (
        not results
        or not l2_errors
        or not l2_errors_area_wt
        or not top_centerlines
        or not bottom_centerlines
    ):
        # Process the data if any of the required files are missing
        with Pool(processes=args.num_procs) as pool:
            mesh_results = pool.map(
                partial(
                    process_surface_results,
                    field_mapping=args.field_mapping,
                    compute_projections=args.plot_aggregate_surface_errors,
                ),
                filenames,
            )

        # Prepare data for saving
        results = {
            "run_idx": [],
            "Cd_true": [],
            "Cd_p_true": [],
            "Cd_f_true": [],
            "Cl_true": [],
            "Cl_p_true": [],
            "Cl_f_true": [],
            "Cd_pred": [],
            "Cd_p_pred": [],
            "Cd_f_pred": [],
            "Cl_pred": [],
            "Cl_p_pred": [],
            "Cl_f_pred": [],
        }
        results_pc = {
            "run_idx": [],
            "Cd_true_pc": [],
            "Cd_p_true_pc": [],
            "Cd_f_true_pc": [],
            "Cl_true_pc": [],
            "Cl_p_true_pc": [],
            "Cl_f_true_pc": [],
            "Cd_pred_pc": [],
            "Cd_p_pred_pc": [],
            "Cd_f_pred_pc": [],
            "Cl_pred_pc": [],
            "Cl_p_pred_pc": [],
            "Cl_f_pred_pc": [],
        }

        l2_errors = {"run_idx": []}
        for key in mesh_results[0]["l2_errors"].keys():
            l2_errors[key] = []
        l2_errors_area_wt = {"run_idx": []}
        for key in mesh_results[0]["l2_errors_area_wt"].keys():
            l2_errors_area_wt[key] = []

        l2_errors_pc = {"run_idx": []}
        if mesh_results[0]["l2_errors_pc"] is not None:
            for key in mesh_results[0]["l2_errors_pc"].keys():
                l2_errors_pc[key] = []
        else:
            l2_errors_pc = None

        top_centerlines = []
        bottom_centerlines = []
        pos_xy_projections = []
        neg_xy_projections = []
        pos_yz_projections = []
        neg_yz_projections = []
        pos_zx_projections = []
        neg_zx_projections = []

        for mesh_result in mesh_results:
            results["run_idx"].append(mesh_result["run_idx"])
            results["Cd_true"].append(mesh_result["Cd_true"])
            results["Cd_p_true"].append(mesh_result["Cd_p_true"])
            results["Cd_f_true"].append(mesh_result["Cd_f_true"])
            results["Cl_true"].append(mesh_result["Cl_true"])
            results["Cl_p_true"].append(mesh_result["Cl_p_true"])
            results["Cl_f_true"].append(mesh_result["Cl_f_true"])
            results["Cd_pred"].append(mesh_result["Cd_pred"])
            results["Cd_p_pred"].append(mesh_result["Cd_p_pred"])
            results["Cd_f_pred"].append(mesh_result["Cd_f_pred"])
            results["Cl_pred"].append(mesh_result["Cl_pred"])
            results["Cl_p_pred"].append(mesh_result["Cl_p_pred"])
            results["Cl_f_pred"].append(mesh_result["Cl_f_pred"])

            if args.pc_results_dir:
                results_pc["run_idx"].append(mesh_result["run_idx"])
                results_pc["Cd_true_pc"].append(mesh_result["Cd_true_pc"])
                results_pc["Cd_p_true_pc"].append(mesh_result["Cd_p_true_pc"])
                results_pc["Cd_f_true_pc"].append(mesh_result["Cd_f_true_pc"])
                results_pc["Cl_true_pc"].append(mesh_result["Cl_true_pc"])
                results_pc["Cl_p_true_pc"].append(mesh_result["Cl_p_true_pc"])
                results_pc["Cl_f_true_pc"].append(mesh_result["Cl_f_true_pc"])
                results_pc["Cd_pred_pc"].append(mesh_result["Cd_pred_pc"])
                results_pc["Cd_p_pred_pc"].append(mesh_result["Cd_p_pred_pc"])
                results_pc["Cd_f_pred_pc"].append(mesh_result["Cd_f_pred_pc"])
                results_pc["Cl_pred_pc"].append(mesh_result["Cl_pred_pc"])
                results_pc["Cl_p_pred_pc"].append(mesh_result["Cl_p_pred_pc"])
                results_pc["Cl_f_pred_pc"].append(mesh_result["Cl_f_pred_pc"])

                l2_errors_pc["run_idx"].append(mesh_result["run_idx"])

            l2_errors["run_idx"].append(mesh_result["run_idx"])
            l2_errors_area_wt["run_idx"].append(mesh_result["run_idx"])

            for key, value in mesh_result["l2_errors"].items():
                l2_errors[key].append(value)
            for key, value in mesh_result["l2_errors_area_wt"].items():
                l2_errors_area_wt[key].append(value)

            if args.pc_results_dir:
                for key, value in mesh_result["l2_errors_pc"].items():
                    l2_errors_pc[key].append(value)

            top_centerlines.append(mesh_result["centerline_top"])
            bottom_centerlines.append(mesh_result["centerline_bottom"])
            pos_xy_projections.append(mesh_result["projection_XY"])
            neg_xy_projections.append(mesh_result["projection_-XY"])
            pos_yz_projections.append(mesh_result["projection_YZ"])
            neg_yz_projections.append(mesh_result["projection_-YZ"])
            pos_zx_projections.append(mesh_result["projection_ZX"])
            neg_zx_projections.append(mesh_result["projection_-ZX"])

        # Save results to CSV
        save_results_to_csv(results, results_csv, results.keys())
        save_results_to_csv(l2_errors, l2_errors_csv, l2_errors.keys())
        save_results_to_csv(
            l2_errors_area_wt, l2_errors_area_wt_csv, l2_errors_area_wt.keys()
        )

        if args.pc_results_dir:
            save_results_to_csv(results_pc, results_pc_csv, results_pc.keys())
            save_results_to_csv(l2_errors_pc, l2_errors_pc_csv, l2_errors_pc.keys())

        # Save centerlines
        save_polydata(top_centerlines, output_dir, "top_centerline", results["run_idx"])
        save_polydata(
            bottom_centerlines, output_dir, "bottom_centerline", results["run_idx"]
        )

        # Save projections
        if args.plot_aggregate_surface_errors:
            save_polydata(
                pos_xy_projections, output_dir, "projection_XY", results["run_idx"]
            )
            save_polydata(
                neg_xy_projections, output_dir, "projection_-XY", results["run_idx"]
            )
            save_polydata(
                pos_yz_projections, output_dir, "projection_YZ", results["run_idx"]
            )
            save_polydata(
                neg_yz_projections, output_dir, "projection_-YZ", results["run_idx"]
            )
            save_polydata(
                pos_zx_projections, output_dir, "projection_ZX", results["run_idx"]
            )
            save_polydata(
                neg_zx_projections, output_dir, "projection_-ZX", results["run_idx"]
            )

    else:
        # Load mesh_results from the saved CSVs
        mesh_results = []
        for i in range(len(results["run_idx"])):
            # print(l2_errors_pc)
            mesh_result = {
                "run_idx": results["run_idx"][i],
                "Cd_true": results["Cd_true"][i],
                "Cd_p_true": results["Cd_p_true"][i],
                "Cd_f_true": results["Cd_f_true"][i],
                "Cl_true": results["Cl_true"][i],
                "Cl_p_true": results["Cl_p_true"][i],
                "Cl_f_true": results["Cl_f_true"][i],
                "Cd_pred": results["Cd_pred"][i],
                "Cd_p_pred": results["Cd_p_pred"][i],
                "Cd_f_pred": results["Cd_f_pred"][i],
                "Cl_pred": results["Cl_pred"][i],
                "Cl_p_pred": results["Cl_p_pred"][i],
                "Cl_f_pred": results["Cl_f_pred"][i],
                "l2_errors": {
                    key: l2_errors[key][i] for key in l2_errors if key != "run_idx"
                },
                "l2_errors_area_wt": {
                    key: l2_errors_area_wt[key][i]
                    for key in l2_errors_area_wt
                    if key != "run_idx"
                },
                "centerline_top": top_centerlines[i],
                "centerline_bottom": bottom_centerlines[i],
            }

            if args.plot_aggregate_surface_errors:
                mesh_result.update(
                    {
                        "projection_XY": pos_xy_projections[i],
                        "projection_-XY": neg_xy_projections[i],
                        "projection_YZ": pos_yz_projections[i],
                        "projection_-YZ": neg_yz_projections[i],
                        "projection_ZX": pos_zx_projections[i],
                        "projection_-ZX": neg_zx_projections[i],
                    }
                )

            if args.pc_results_dir:
                mesh_result.update(
                    {
                        "Cd_true_pc": results_pc["Cd_true_pc"][i],
                        "Cd_p_true_pc": results_pc["Cd_p_true_pc"][i],
                        "Cd_f_true_pc": results_pc["Cd_f_true_pc"][i],
                        "Cl_true_pc": results_pc["Cl_true_pc"][i],
                        "Cl_p_true_pc": results_pc["Cl_p_true_pc"][i],
                        "Cl_f_true_pc": results_pc["Cl_f_true_pc"][i],
                        "Cd_pred_pc": results_pc["Cd_pred_pc"][i],
                        "Cd_p_pred_pc": results_pc["Cd_p_pred_pc"][i],
                        "Cd_f_pred_pc": results_pc["Cd_f_pred_pc"][i],
                        "Cl_pred_pc": results_pc["Cl_pred_pc"][i],
                        "Cl_p_pred_pc": results_pc["Cl_p_pred_pc"][i],
                        "Cl_f_pred_pc": results_pc["Cl_f_pred_pc"][i],
                        "l2_errors_pc": {
                            key: l2_errors_pc[key][i]
                            for key in l2_errors_pc
                            if key != "run_idx"
                        },
                    }
                )

            mesh_results.append(mesh_result)

    # combine results
    true_data_dict = {"Cd": [], "Cl": []}
    pred_data_dict = {"Cd": [], "Cl": []}
    idx_dict = {"Cd": [], "Cl": []}
    if args.pc_results_dir:
        true_data_dict_pc = {"Cd": [], "Cl": []}
        pred_data_dict_pc = {"Cd": [], "Cl": []}

    top_centerlines = []
    bottom_centerlines = []

    mean_l2_errors = {}
    for key in mesh_results[0]["l2_errors"].keys():
        mean_l2_errors[key] = []

    mean_area_wt_l2_errors = {}
    for key in mesh_results[0]["l2_errors_area_wt"].keys():
        mean_area_wt_l2_errors[key] = []

    if args.pc_results_dir:
        mean_l2_errors_pc = {}
        for key in mesh_results[0]["l2_errors_pc"].keys():
            mean_l2_errors_pc[key] = []
    else:
        mean_l2_errors_pc = None

    for mesh_result in mesh_results:
        true_data_dict["Cd"].append(mesh_result["Cd_true"])
        true_data_dict["Cl"].append(mesh_result["Cl_true"])
        pred_data_dict["Cd"].append(mesh_result["Cd_pred"])
        pred_data_dict["Cl"].append(mesh_result["Cl_pred"])
        idx_dict["Cd"].append(mesh_result["run_idx"])
        idx_dict["Cl"].append(mesh_result["run_idx"])
        top_centerlines.append(mesh_result["centerline_top"])
        bottom_centerlines.append(mesh_result["centerline_bottom"])
        if args.plot_aggregate_surface_errors:
            pos_xy_projections.append(mesh_result["projection_XY"])
            neg_xy_projections.append(mesh_result["projection_-XY"])
            pos_yz_projections.append(mesh_result["projection_YZ"])
            neg_yz_projections.append(mesh_result["projection_-YZ"])
            pos_zx_projections.append(mesh_result["projection_ZX"])
            neg_zx_projections.append(mesh_result["projection_-ZX"])

        for key, value in mesh_result["l2_errors"].items():
            mean_l2_errors[key].append(value)
        for key, value in mesh_result["l2_errors_area_wt"].items():
            mean_area_wt_l2_errors[key].append(value)
        if args.pc_results_dir:
            true_data_dict_pc["Cd"].append(mesh_result["Cd_true_pc"])
            true_data_dict_pc["Cl"].append(mesh_result["Cl_true_pc"])
            pred_data_dict_pc["Cd"].append(mesh_result["Cd_pred_pc"])
            pred_data_dict_pc["Cl"].append(mesh_result["Cl_pred_pc"])
            for key, value in mesh_result["l2_errors_pc"].items():
                mean_l2_errors_pc[key].append(value)

    for key, value in mean_l2_errors.items():
        mean_l2_errors[key] = np.mean(np.array(value))

    for key, value in mean_area_wt_l2_errors.items():
        mean_area_wt_l2_errors[key] = np.mean(np.array(value))

    if args.pc_results_dir:
        for key, value in mean_l2_errors_pc.items():
            mean_l2_errors_pc[key] = np.mean(np.array(value))

    fig = plot_design_scatter(
        true_data_dict,
        pred_data_dict,
        figsize=(15, 6),
        regression_line_kwargs={"color": "black", "linestyle": "--"},
        title_kwargs={"fontsize": 10},
    )[0]
    fig.savefig(f"./{output_dir}/design_scatter_plot.png")

    fig = plot_design_trend(
        true_data_dict,
        pred_data_dict,
        idx_dict,
        figsize=(15, 6),
        true_line_kwargs={"color": "red"},
        pred_line_kwargs={"color": "green"},
        title_kwargs={"fontsize": 10},
    )[0]
    fig.savefig(f"./{output_dir}/design_trend_plot.png")

    if args.pc_results_dir:
        fig = plot_design_scatter(
            true_data_dict,  # plot against the true data from simulation mesh
            pred_data_dict_pc,
            figsize=(15, 6),
            regression_line_kwargs={"color": "black", "linestyle": "--"},
            title_kwargs={"fontsize": 10},
        )[0]
        fig.savefig(f"./{output_dir}/design_scatter_pc_plot.png")

        fig = plot_design_trend(
            true_data_dict,  # plot against the true data from simulation mesh
            pred_data_dict_pc,
            idx_dict,
            figsize=(15, 6),
            true_line_kwargs={"color": "red"},
            pred_line_kwargs={"color": "green"},
            title_kwargs={"fontsize": 10},
        )[0]
        fig.savefig(f"./{output_dir}/design_trend_pc_plot.png")

    fig = plot_line(
        top_centerlines,
        plot_coord="x",
        field_true=args.field_mapping["p"],
        field_pred=args.field_mapping["pPred"],
        normalize_factor=(1.0 * 38.889**2) / 2,
        true_line_kwargs={"color": "red", "alpha": 1 / len(top_centerlines)},
        pred_line_kwargs={"color": "green", "alpha": 1 / len(top_centerlines)},
        mean_true_line_kwargs={"color": "red", "label": "Mean True"},
        mean_pred_line_kwargs={"color": "green", "label": "Mean Pred"},
        title_kwargs={"fontsize": 12},
        figsize=(8, 6),
        xlabel="X Coordinate",
        ylabel="p / U_ref^2",
    )
    fig.savefig(f"./{output_dir}/top_centerline.png")

    fig = plot_line(
        bottom_centerlines,
        plot_coord="x",
        field_true=args.field_mapping["p"],
        field_pred=args.field_mapping["pPred"],
        normalize_factor=(1.0 * 38.889**2) / 2,
        true_line_kwargs={"color": "red", "alpha": 1 / len(bottom_centerlines)},
        pred_line_kwargs={"color": "green", "alpha": 1 / len(bottom_centerlines)},
        mean_true_line_kwargs={"color": "red", "label": "Mean True"},
        mean_pred_line_kwargs={"color": "green", "label": "Mean Pred"},
        title_kwargs={"fontsize": 12},
        figsize=(8, 6),
        xlabel="X Coordinate",
        ylabel="p / U_ref^2",
    )
    fig.savefig(f"./{output_dir}/bottom_centerline.png")

    if args.plot_aggregate_surface_errors:
        fig = plot_projections_hexbin(pos_xy_projections, "p_error", "XY")
        fig.savefig(f"./{output_dir}/hexbin_p_error_XY.png", dpi=300)
        fig = plot_projections_hexbin(pos_xy_projections, "wallShearStress_error", "XY")
        fig.savefig(f"./{output_dir}/hexbin_wallShearStress_error_XY.png", dpi=300)
        fig = plot_projections_hexbin(pos_yz_projections, "p_error", "YZ")
        fig.savefig(f"./{output_dir}/hexbin_p_error_YZ.png", dpi=300)
        fig = plot_projections_hexbin(pos_yz_projections, "wallShearStress_error", "YZ")
        fig.savefig(f"./{output_dir}/hexbin_wallShearStress_error_YZ.png", dpi=300)
        fig = plot_projections_hexbin(pos_zx_projections, "p_error", "ZX")
        fig.savefig(f"./{output_dir}/hexbin_p_error_ZX.png", dpi=300)
        fig = plot_projections_hexbin(pos_zx_projections, "wallShearStress_error", "ZX")
        fig.savefig(f"./{output_dir}/hexbin_wallShearStress_error_ZX.png", dpi=300)

    for key, value in mean_l2_errors.items():
        print(f"L2 Errors for {key}: {value}")

    for key, value in mean_area_wt_l2_errors.items():
        print(f"Area weighted L2 Errors for {key}: {value}")

    if args.pc_results_dir:
        for key, value in mean_l2_errors_pc.items():
            print(f"L2 Errors for PC, {key}: {value}")

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
                    plot_surface_results,
                    field_mapping=args.field_mapping,
                    output_dir=args.output_dir,
                ),
                plot_filenames,
            )
