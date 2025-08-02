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

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from scipy.interpolate import interp1d
import vtk
from scipy.spatial import cKDTree

pv.start_xvfb()


def plot_field_comparisons(
    data,
    true_fields,
    pred_fields,
    view="xy",
    dtype="cell",
    plot_vector_components=False,
    window_size=[2560, 3840],
    view_negative=False,
    **kwargs,
):
    """Helper function to compare fields"""
    assert len(true_fields) == len(
        pred_fields
    ), "You must provide same number of true and pred fields"

    # identify vector and scalar quantities
    field_type = {}
    for field in true_fields:
        arr = data.get_array(field, preference=dtype)
        if len(arr.shape) == 1:
            field_type[field] = "scalar"
        else:
            field_type[field] = "vector"

    num_vector_fields = sum(1 for key, value in field_type.items() if value == "vector")
    num_scalar_fields = sum(1 for key, value in field_type.items() if value == "scalar")

    if plot_vector_components:
        plotter = pv.Plotter(
            shape=((num_vector_fields * 3 + num_scalar_fields), 3),
            window_size=window_size,
            off_screen=True,
        )
    else:
        plotter = pv.Plotter(
            shape=((num_vector_fields + num_scalar_fields), 3),
            window_size=window_size,
            off_screen=True,
        )

    cmap = plt.get_cmap(kwargs.get("cmap", "viridis"), lut=kwargs.get("lut", None))

    plot_idx = 0
    for true_field, pred_field in zip(true_fields, pred_fields):
        if field_type[true_field] == "scalar":
            true_data = data.get_array(true_field, preference=dtype)
            pred_data = data.get_array(pred_field, preference=dtype)
            mean = np.mean(true_data)
            std = np.std(true_data)
            vmin = mean - 2 * std
            vmax = mean + 2 * std

            error = np.abs(true_data - pred_data)

            if dtype == "cell":
                data.cell_data[f"{true_field}_error"] = error
            elif dtype == "point":
                data.point_data[f"{true_field}_error"] = error

            err_mean = np.mean(error)
            err_std = np.std(error)
            err_vmin = err_mean - 2 * err_std
            err_vmax = err_mean + 2 * err_std

            plotter.subplot(plot_idx, 0)
            data.set_active_scalars(true_field, preference=dtype)

            if np.log10(np.abs(vmin)) < -1 or np.log10(np.abs(vmax)) < -1:
                scalar_bar_args = {**kwargs.get("scalar_bar_args", {}), "fmt": "%.2g"}
            else:
                scalar_bar_args = {**kwargs.get("scalar_bar_args", {}), "fmt": "%.1f"}

            if np.log10(np.abs(err_vmin)) < -1 or np.log10(np.abs(err_vmax)) < -1:
                err_scalar_bar_args = {
                    **kwargs.get("scalar_bar_args", {}),
                    "fmt": "%.2g",
                }
            else:
                err_scalar_bar_args = {
                    **kwargs.get("scalar_bar_args", {}),
                    "fmt": "%.1f",
                }

            plotter.add_mesh(
                data,
                copy_mesh=True,
                cmap=cmap,
                clim=(vmin, vmax),
                scalar_bar_args=scalar_bar_args,
            )
            plotter.add_text(f"{true_field}")
            if view == "xy":
                plotter.view_xy(negative=view_negative)
            elif view == "yz":
                plotter.view_yz(negative=view_negative)
            elif view == "xz":
                plotter.view_xz(negative=view_negative)

            plotter.subplot(plot_idx, 1)
            data.set_active_scalars(pred_field, preference=dtype)
            plotter.add_mesh(
                data,
                copy_mesh=True,
                cmap=cmap,
                clim=(vmin, vmax),
                scalar_bar_args=scalar_bar_args,
            )
            plotter.add_text(f"{pred_field}")
            if view == "xy":
                plotter.view_xy(negative=view_negative)
            elif view == "yz":
                plotter.view_yz(negative=view_negative)
            elif view == "xz":
                plotter.view_xz(negative=view_negative)

            plotter.subplot(plot_idx, 2)
            data.set_active_scalars(f"{true_field}_error", preference=dtype)
            plotter.add_mesh(
                data,
                copy_mesh=True,
                cmap=cmap,
                clim=(err_vmin, err_vmax),
                scalar_bar_args=err_scalar_bar_args,
            )
            plotter.add_text(f"|{true_field} - {pred_field}|")
            if view == "xy":
                plotter.view_xy(negative=view_negative)
            elif view == "yz":
                plotter.view_yz(negative=view_negative)
            elif view == "xz":
                plotter.view_xz(negative=view_negative)

            plot_idx += 1

        if field_type[true_field] == "vector":
            if plot_vector_components:
                for i, component in enumerate(["x", "y", "z"]):
                    true_data = data.get_array(true_field, preference=dtype)[:, i]
                    pred_data = data.get_array(pred_field, preference=dtype)[:, i]
                    mean = np.mean(true_data)
                    std = np.std(true_data)
                    vmin = mean - 2 * std
                    vmax = mean + 2 * std

                    error = np.abs(true_data - pred_data)

                    if dtype == "cell":
                        data.cell_data[f"{true_field}_{component}"] = true_data
                        data.cell_data[f"{pred_field}_{component}"] = pred_data
                        data.cell_data[f"{true_field}_{component}_error"] = error
                    elif dtype == "point":
                        data.point_data[f"{true_field}_{component}"] = true_data
                        data.point_data[f"{pred_field}_{component}"] = pred_data
                        data.point_data[f"{true_field}_{component}_error"] = error

                    err_mean = np.mean(error)
                    err_std = np.std(error)
                    err_vmin = err_mean - 2 * err_std
                    err_vmax = err_mean + 2 * err_std

                    if np.log10(np.abs(vmin)) < -1 or np.log10(np.abs(vmax)) < -1:
                        scalar_bar_args = {
                            **kwargs.get("scalar_bar_args", {}),
                            "fmt": "%.2g",
                        }
                    else:
                        scalar_bar_args = {
                            **kwargs.get("scalar_bar_args", {}),
                            "fmt": "%.1f",
                        }

                    if (
                        np.log10(np.abs(err_vmin)) < -1
                        or np.log10(np.abs(err_vmax)) < -1
                    ):
                        err_scalar_bar_args = {
                            **kwargs.get("scalar_bar_args", {}),
                            "fmt": "%.2g",
                        }
                    else:
                        err_scalar_bar_args = {
                            **kwargs.get("scalar_bar_args", {}),
                            "fmt": "%.1f",
                        }

                    plotter.subplot(plot_idx, 0)
                    data.set_active_scalars(
                        f"{true_field}_{component}", preference=dtype
                    )
                    plotter.add_mesh(
                        data,
                        copy_mesh=True,
                        cmap=cmap,
                        clim=(vmin, vmax),
                        scalar_bar_args=scalar_bar_args,
                    )
                    plotter.add_text(f"{true_field}_{component}")
                    if view == "xy":
                        plotter.view_xy(negative=view_negative)
                    elif view == "yz":
                        plotter.view_yz(negative=view_negative)
                    elif view == "xz":
                        plotter.view_xz(negative=view_negative)

                    plotter.subplot(plot_idx, 1)
                    data.set_active_scalars(
                        f"{pred_field}_{component}", preference=dtype
                    )
                    plotter.add_mesh(
                        data,
                        copy_mesh=True,
                        cmap=cmap,
                        clim=(vmin, vmax),
                        scalar_bar_args=scalar_bar_args,
                    )
                    plotter.add_text(f"{pred_field}_{component}")
                    if view == "xy":
                        plotter.view_xy(negative=view_negative)
                    elif view == "yz":
                        plotter.view_yz(negative=view_negative)
                    elif view == "xz":
                        plotter.view_xz(negative=view_negative)

                    plotter.subplot(plot_idx, 2)
                    data.set_active_scalars(
                        f"{true_field}_{component}_error", preference=dtype
                    )
                    plotter.add_mesh(
                        data,
                        copy_mesh=True,
                        cmap=cmap,
                        clim=(err_vmin, err_vmax),
                        scalar_bar_args=err_scalar_bar_args,
                    )
                    plotter.add_text(
                        f"|{true_field}_{component} - {pred_field}_{component}|"
                    )
                    if view == "xy":
                        plotter.view_xy(negative=view_negative)
                    elif view == "yz":
                        plotter.view_yz(negative=view_negative)
                    elif view == "xz":
                        plotter.view_xz(negative=view_negative)

                    plot_idx += 1
            else:
                true_data = data.get_array(true_field, preference=dtype)
                pred_data = data.get_array(pred_field, preference=dtype)

                true_magnitude = np.linalg.norm(true_data, axis=1)
                pred_magnitude = np.linalg.norm(pred_data, axis=1)
                mean = np.mean(true_magnitude)
                std = np.std(true_magnitude)
                vmin = mean - 2 * std
                vmax = mean + 2 * std

                error = np.abs(true_data - pred_data)
                error_magnitude = np.linalg.norm(error, axis=1)
                err_mean = np.mean(error_magnitude)
                err_std = np.std(error_magnitude)
                err_vmin = err_mean - 2 * err_std
                err_vmax = err_mean + 2 * err_std

                if np.log10(np.abs(vmin)) < -1 or np.log10(np.abs(vmax)) < -1:
                    scalar_bar_args = {
                        **kwargs.get("scalar_bar_args", {}),
                        "fmt": "%.2g",
                    }
                else:
                    scalar_bar_args = {
                        **kwargs.get("scalar_bar_args", {}),
                        "fmt": "%.1f",
                    }

                if np.log10(np.abs(err_vmin)) < -1 or np.log10(np.abs(err_vmax)) < -1:
                    err_scalar_bar_args = {
                        **kwargs.get("scalar_bar_args", {}),
                        "fmt": "%.2g",
                    }
                else:
                    err_scalar_bar_args = {
                        **kwargs.get("scalar_bar_args", {}),
                        "fmt": "%.1f",
                    }

                if dtype == "cell":
                    data.cell_data[f"{true_field}_mag"] = true_magnitude
                    data.cell_data[f"{pred_field}_mag"] = pred_magnitude
                    data.cell_data[f"{true_field}_error"] = error_magnitude
                elif dtype == "point":
                    data.point_data[f"{true_field}_mag"] = true_magnitude
                    data.point_data[f"{pred_field}_mag"] = pred_magnitude
                    data.point_data[f"{true_field}_error"] = error_magnitude

                plotter.subplot(plot_idx, 0)
                data.set_active_scalars(f"{true_field}_mag", preference=dtype)
                plotter.add_mesh(
                    data,
                    copy_mesh=True,
                    cmap=cmap,
                    clim=(vmin, vmax),
                    scalar_bar_args=scalar_bar_args,
                )
                plotter.add_text(f"{true_field}")
                if view == "xy":
                    plotter.view_xy(negative=view_negative)
                elif view == "yz":
                    plotter.view_yz(negative=view_negative)
                elif view == "xz":
                    plotter.view_xz(negative=view_negative)

                plotter.subplot(plot_idx, 1)
                data.set_active_scalars(f"{pred_field}_mag", preference=dtype)
                plotter.add_mesh(
                    data,
                    copy_mesh=True,
                    cmap=cmap,
                    clim=(vmin, vmax),
                    scalar_bar_args=scalar_bar_args,
                )
                plotter.add_text(f"{pred_field}")
                if view == "xy":
                    plotter.view_xy(negative=view_negative)
                elif view == "yz":
                    plotter.view_yz(negative=view_negative)
                elif view == "xz":
                    plotter.view_xz(negative=view_negative)

                plotter.subplot(plot_idx, 2)
                data.set_active_scalars(f"{true_field}_error", preference=dtype)
                plotter.add_mesh(
                    data,
                    copy_mesh=True,
                    cmap=cmap,
                    clim=(err_vmin, err_vmax),
                    scalar_bar_args=err_scalar_bar_args,
                )
                plotter.add_text(f"|{true_field} - {pred_field}|")
                if view == "xy":
                    plotter.view_xy(negative=view_negative)
                elif view == "yz":
                    plotter.view_yz(negative=view_negative)
                elif view == "xz":
                    plotter.view_xz(negative=view_negative)

                plot_idx += 1

    return plotter


def plot_fields(
    data,
    fields,
    view="xy",
    dtype="cell",
    plot_vector_components=False,
    window_size=[2560, 3840],
    view_negative=False,
    **kwargs,
):
    """Helper function to plot fields"""

    # identify vector and scalar quantities
    field_type = {}
    for field in fields:
        arr = data.get_array(field, preference=dtype)
        if len(arr.shape) == 1:
            field_type[field] = "scalar"
        else:
            field_type[field] = "vector"

    num_vector_fields = sum(1 for key, value in field_type.items() if value == "vector")
    num_scalar_fields = sum(1 for key, value in field_type.items() if value == "scalar")

    if plot_vector_components:
        plotter = pv.Plotter(
            shape=((num_vector_fields * 3 + num_scalar_fields), 1),
            window_size=window_size,
            off_screen=True,
        )
    else:
        plotter = pv.Plotter(
            shape=((num_vector_fields + num_scalar_fields), 1),
            window_size=window_size,
            off_screen=True,
        )

    cmap = plt.get_cmap(kwargs.get("cmap", "viridis"), lut=kwargs.get("lut", None))

    plot_idx = 0
    for field in fields:
        if field_type[field] == "scalar":
            field_data = data.get_array(field, preference=dtype)
            mean = np.mean(field_data)
            std = np.std(field_data)
            vmin = mean - 2 * std
            vmax = mean + 2 * std

            plotter.subplot(plot_idx, 0)
            data.set_active_scalars(field, preference=dtype)

            if np.log10(np.abs(vmin)) < -1 or np.log10(np.abs(vmax)) < -1:
                scalar_bar_args = {**kwargs.get("scalar_bar_args", {}), "fmt": "%.2g"}
            else:
                scalar_bar_args = {**kwargs.get("scalar_bar_args", {}), "fmt": "%.1f"}

            plotter.add_mesh(
                data,
                copy_mesh=True,
                cmap=cmap,
                clim=(vmin, vmax),
                scalar_bar_args=scalar_bar_args,
            )
            plotter.add_text(f"{field}")
            if view == "xy":
                plotter.view_xy(negative=view_negative)
            elif view == "yz":
                plotter.view_yz(negative=view_negative)
            elif view == "xz":
                plotter.view_xz(negative=view_negative)

            plot_idx += 1

        if field_type[field] == "vector":
            if plot_vector_components:
                for i, component in enumerate(["x", "y", "z"]):
                    field_data = data.get_array(field, preference=dtype)[:, i]
                    mean = np.mean(field_data)
                    std = np.std(field_data)
                    vmin = mean - 2 * std
                    vmax = mean + 2 * std

                    if dtype == "cell":
                        data.cell_data[f"{field}_{component}"] = field_data
                    elif dtype == "point":
                        data.point_data[f"{field}_{component}"] = field_data

                    if np.log10(np.abs(vmin)) < -1 or np.log10(np.abs(vmax)) < -1:
                        scalar_bar_args = {
                            **kwargs.get("scalar_bar_args", {}),
                            "fmt": "%.2g",
                        }
                    else:
                        scalar_bar_args = {
                            **kwargs.get("scalar_bar_args", {}),
                            "fmt": "%.1f",
                        }

                    plotter.subplot(plot_idx, 0)
                    data.set_active_scalars(f"{field}_{component}", preference=dtype)
                    plotter.add_mesh(
                        data,
                        copy_mesh=True,
                        cmap=cmap,
                        clim=(vmin, vmax),
                        scalar_bar_args=scalar_bar_args,
                    )
                    plotter.add_text(f"{field}_{component}")
                    if view == "xy":
                        plotter.view_xy(negative=view_negative)
                    elif view == "yz":
                        plotter.view_yz(negative=view_negative)
                    elif view == "xz":
                        plotter.view_xz(negative=view_negative)

                    plot_idx += 1
            else:
                field_data = data.get_array(field, preference=dtype)
                field_magnitude = np.linalg.norm(field_data, axis=1)
                mean = np.mean(field_magnitude)
                std = np.std(field_magnitude)
                vmin = mean - 2 * std
                vmax = mean + 2 * std

                if np.log10(np.abs(vmin)) < -1 or np.log10(np.abs(vmax)) < -1:
                    scalar_bar_args = {
                        **kwargs.get("scalar_bar_args", {}),
                        "fmt": "%.2g",
                    }
                else:
                    scalar_bar_args = {
                        **kwargs.get("scalar_bar_args", {}),
                        "fmt": "%.1f",
                    }

                if dtype == "cell":
                    data.cell_data[f"{field}_mag"] = field_magnitude
                elif dtype == "point":
                    data.point_data[f"{field}_mag"] = field_magnitude

                plotter.subplot(plot_idx, 0)
                data.set_active_scalars(f"{field}_mag", preference=dtype)
                plotter.add_mesh(
                    data,
                    copy_mesh=True,
                    cmap=cmap,
                    clim=(vmin, vmax),
                    scalar_bar_args=scalar_bar_args,
                )
                plotter.add_text(f"{field}")
                if view == "xy":
                    plotter.view_xy(negative=view_negative)
                elif view == "yz":
                    plotter.view_yz(negative=view_negative)
                elif view == "xz":
                    plotter.view_xz(negative=view_negative)

                plot_idx += 1

    return plotter


def plot_streamlines(
    true_streamlines,
    pred_streamlines,
    geometry=None,
    view="xy",
    window_size=[3840, 1080],
    **kwargs,
):
    """Helper function to plot streamlines"""
    plotter = pv.Plotter(shape=(1, 2), window_size=window_size, off_screen=True)

    plotter.subplot(0, 0)
    if geometry is not None:
        plotter.add_mesh(
            geometry,
            color=kwargs.get("geometry_color", "lightgrey"),
            opacity=kwargs.get("geometry_opacity", 1.0),
        )  # Add the original mesh

    plotter.add_mesh(
        true_streamlines,
        color=kwargs.get("color", "black"),
        line_width=kwargs.get("line_width", 1.5),
    )  # Add streamlines
    plotter.add_text("Streamlines (True)")
    if view == "xy":
        plotter.view_xy()
    elif view == "yz":
        plotter.view_yz()
    elif view == "xz":
        plotter.view_xz()

    plotter.subplot(0, 1)
    if geometry is not None:
        plotter.add_mesh(
            geometry,
            color=kwargs.get("geometry_color", "lightgrey"),
            opacity=kwargs.get("geometry_opacity", 1.0),
        )  # Add the original mesh

    plotter.add_mesh(
        pred_streamlines,
        color=kwargs.get("color", "black"),
        line_width=kwargs.get("line_width", 1.5),
    )  # Add streamlines
    plotter.add_text("Streamlines (Pred)")
    if view == "xy":
        plotter.view_xy()
    elif view == "yz":
        plotter.view_yz()
    elif view == "xz":
        plotter.view_xz()

    return plotter


def plot_design_scatter(true_data_dict, pred_data_dict, **kwargs):
    """Helper function to plot predicted vs. true results and compute R2 scores."""
    # Assumes dicts of following structure
    # true_data_dict = {"Force": [1.0, 2.0, 3.0], "Force P": [1.0, 2.0, 3.0]}
    # pred_data_dict = {"Force": [1.1, 2.2, 3.3], "Force P": [1.3, 2.4, 3.5]}

    assert len(true_data_dict.keys()) == len(
        pred_data_dict.keys()
    ), "Dicts have unequal number of keys"

    fig, axs = plt.subplots(
        1,
        len(true_data_dict.keys()),
        figsize=kwargs.get("figsize", (5 * len(true_data_dict.keys()), 5)),
    )

    r_squared_dict = {}
    for i, key in enumerate(true_data_dict.keys()):
        axs[i].scatter(
            true_data_dict[key], pred_data_dict[key], **kwargs.get("scatter_kwargs", {})
        )
        r_squared = r2_score(
            np.array(true_data_dict[key]), np.array(pred_data_dict[key])
        )
        r_squared_dict[key] = r_squared

        axs[i].set_xlabel(f"{key} True")
        axs[i].set_ylabel(f"{key} Pred")

        # plot regression line
        x = np.linspace(
            np.min(np.array(true_data_dict[key])),
            np.max(np.array(true_data_dict[key])),
            10,
        )
        y = x
        axs[i].plot(x, y, **kwargs.get("regression_line_kwargs", {}))

        global_min = np.min(np.array(true_data_dict[key]))
        global_max = np.max(np.array(true_data_dict[key]))

        axs[i].set_xlim([global_min * 0.95, global_max * 1.05])
        axs[i].set_ylim([global_min * 0.95, global_max * 1.05])
        axs[i].set_aspect("equal", adjustable="box")

        axs[i].grid(True, linestyle="--", alpha=0.5)

        axs[i].set_title(
            f"{key}. R2: {r_squared:.4f}", **kwargs.get("title_kwargs", {})
        )

    return fig, r_squared_dict


def plot_design_trend(true_data_dict, pred_data_dict, idx_dict, **kwargs):
    """Helper function to plot the directional changes between predicted and true results
    and compare using statistics like Spearman coefficient, Mean Absolute Error, and Max
    Absolute Error.
    """
    assert len(true_data_dict.keys()) == len(
        pred_data_dict.keys()
    ), "Dicts have unequal number of keys"

    fig, axs = plt.subplots(
        1,
        len(true_data_dict.keys()),
        figsize=kwargs.get("figsize", (8 * len(true_data_dict.keys()), 6)),
    )

    spearman_coeff_dict = {}
    mae_dict = {}
    max_err_dict = {}
    min_err_dict = {}

    sorted_true_data_dict = {}
    sorted_pred_data_dict = {}
    sorted_idx_dict = {}
    for i, key in enumerate(true_data_dict.keys()):
        sorted_pairs = sorted(
            zip(true_data_dict[key], pred_data_dict[key], idx_dict[key]),
            key=lambda x: x[0],
        )
        sorted_true_list = [item for item, _, _ in sorted_pairs]
        sorted_pred_list = [item for _, item, _ in sorted_pairs]
        sorted_idx_list = [str(item) for _, _, item in sorted_pairs]

        sorted_true_data_dict[key] = sorted_true_list
        sorted_pred_data_dict[key] = sorted_pred_list
        sorted_idx_dict[key] = sorted_idx_list

        correlation, _ = spearmanr(sorted_true_list, sorted_pred_list)
        max_error = np.max(
            np.abs(np.array(sorted_true_list) - np.array(sorted_pred_list))
        )
        min_error = np.max(
            np.abs(np.array(sorted_true_list) - np.array(sorted_pred_list))
        )
        mae = np.mean(np.abs(np.array(sorted_true_list) - np.array(sorted_pred_list)))

        spearman_coeff_dict[key] = correlation
        mae_dict = mae
        max_err_dict = max_error
        min_err_dict = min_error

        global_min = np.min(np.array(sorted_true_list))
        global_max = np.max(np.array(sorted_true_list))

        axs[i].plot(
            sorted_idx_list,
            sorted_true_list,
            label=f"{key} (True)",
            **kwargs.get("true_line_kwargs", {}),
        )
        axs[i].plot(
            sorted_idx_list,
            sorted_pred_list,
            label=f"{key} (Pred)",
            **kwargs.get("pred_line_kwargs", {}),
        )

        axs[i].set_ylim([global_min * 0.95, global_max * 1.05])
        axs[i].tick_params(axis="x", labelrotation=90, labelsize=4)
        axs[i].grid(True, linestyle="--", alpha=0.5)
        axs[i].set_xlabel("IDs")
        axs[i].set_ylabel(f"{key}")

        axs[i].set_title(
            f"Trend.\nSpearman Corr: {correlation:.2e}. Mean Abs. Error: {mae:.2e}. Max Abs. Error: {max_error:.2e}",
            **kwargs.get("title_kwargs", {}),
        )

        axs[i].legend()

    return (
        fig,
        sorted_true_data_dict,
        sorted_pred_data_dict,
        sorted_idx_dict,
        spearman_coeff_dict,
        mae_dict,
        max_err_dict,
        min_err_dict,
    )


def plot_line(
    lines,
    plot_coord="x",
    field_true="UMeanTrim",
    field_pred="UMeanTrimPred",
    normalize_factor=38.889,
    coord_trim=(None, None),
    field_trim=(None, None),
    flip=False,
    **kwargs,
):
    """Helper function for line plots"""
    fig, ax = plt.subplots()

    if plot_coord not in ["x", "y", "z"]:
        raise ValueError(f"Unsupported plot_coord: {plot_coord}")

    if isinstance(lines, list):
        min_points = []
        min_pt = []
        max_pt = []

        for line in lines:
            if plot_coord == "x":
                pts = line.points[:, 0]
            elif plot_coord == "y":
                pts = line.points[:, 1]
            elif plot_coord == "z":
                pts = line.points[:, 2]

            line = line.cell_data_to_point_data()
            min_points.append(line.points.shape[0])
            min_pt.append(np.min(pts))
            max_pt.append(np.max(pts))

            true_data = line.point_data[field_true]
            pred_data = line.point_data[field_pred]

            if len(true_data.shape) > 1:
                true_data = np.linalg.norm(true_data, axis=1)
                pred_data = np.linalg.norm(pred_data, axis=1)

            sorted_indices = np.argsort(pts)
            pts = pts[sorted_indices]
            true_data = true_data[sorted_indices]
            pred_data = pred_data[sorted_indices]

            if coord_trim[0] is not None and coord_trim[1] is not None:
                mask = (pts >= coord_trim[0]) & (pts <= coord_trim[1])
                pts = pts[mask]
                true_data = true_data[mask]
                pred_data = pred_data[mask]

            if flip:
                ax.plot(
                    true_data / normalize_factor,
                    pts,
                    **kwargs.get("true_line_kwargs", {}),
                )
                ax.plot(
                    pred_data / normalize_factor,
                    pts,
                    **kwargs.get("pred_line_kwargs", {}),
                )
                if field_trim[0] is not None and field_trim[1] is not None:
                    ax.set_xlim(field_trim[0], field_trim[1])
                    ax.set_ylim(coord_trim[0], coord_trim[1])
            else:
                ax.plot(
                    pts,
                    true_data / normalize_factor,
                    **kwargs.get("true_line_kwargs", {}),
                )
                ax.plot(
                    pts,
                    pred_data / normalize_factor,
                    **kwargs.get("pred_line_kwargs", {}),
                )
                if field_trim[0] is not None and field_trim[1] is not None:
                    ax.set_ylim(field_trim[0], field_trim[1])
                    ax.set_xlim(coord_trim[0], coord_trim[1])

        min_points = np.min(np.array(min_points))
        min_pt = np.max(np.array(min_pt))  # maximum, minimum coord
        max_pt = np.min(np.array(max_pt))  # minimum, maximum coord
        mean_points = np.linspace(min_pt, max_pt, min_points)

        filtered_true_fields = []
        filtered_pred_fields = []
        for line in lines:
            if plot_coord == "x":
                pts = line.points[:, 0]
            elif plot_coord == "y":
                pts = line.points[:, 1]
            elif plot_coord == "z":
                pts = line.points[:, 2]

            line = line.cell_data_to_point_data()
            true_data = line.point_data[field_true]
            pred_data = line.point_data[field_pred]
            if len(true_data.shape) > 1:
                true_data = np.linalg.norm(true_data, axis=1)
                pred_data = np.linalg.norm(pred_data, axis=1)

            interp_func_true = interp1d(pts, true_data, kind="nearest")
            interp_func_pred = interp1d(pts, pred_data, kind="nearest")
            filtered_true_fields.append(interp_func_true(mean_points))
            filtered_pred_fields.append(interp_func_pred(mean_points))

        mean_true_fields = np.mean(np.stack(filtered_true_fields, axis=0), axis=0)
        mean_pred_fields = np.mean(np.stack(filtered_pred_fields, axis=0), axis=0)

        if len(mean_true_fields.shape) > 1:
            mean_true_fields = np.linalg.norm(mean_true_fields, axis=1)
            mean_pred_fields = np.linalg.norm(mean_pred_fields, axis=1)

        sorted_indices = np.argsort(mean_points)
        mean_points = mean_points[sorted_indices]
        mean_true_fields = mean_true_fields[sorted_indices]
        mean_pred_fields = mean_pred_fields[sorted_indices]

        max_error = np.max(
            np.abs((mean_true_fields - mean_pred_fields)[sorted_indices])
        )
        mae = np.mean(np.abs((mean_true_fields - mean_pred_fields)[sorted_indices]))

        if coord_trim[0] is not None and coord_trim[1] is not None:
            mask = (mean_points >= coord_trim[0]) & (mean_points <= coord_trim[1])
            mean_points = mean_points[mask]
            mean_true_fields = mean_true_fields[mask]
            mean_pred_fields = mean_pred_fields[mask]

        if flip:
            ax.plot(
                mean_true_fields / normalize_factor,
                mean_points,
                **kwargs.get("mean_true_line_kwargs", {}),
            )
            ax.plot(
                mean_pred_fields / normalize_factor,
                mean_points,
                **kwargs.get("mean_pred_line_kwargs", {}),
            )
            if field_trim[0] is not None and field_trim[1] is not None:
                ax.set_xlim(field_trim[0], field_trim[1])
                ax.set_ylim(coord_trim[0], coord_trim[1])
        else:
            ax.plot(
                mean_points,
                mean_true_fields / normalize_factor,
                **kwargs.get("mean_true_line_kwargs", {}),
            )
            ax.plot(
                mean_points,
                mean_pred_fields / normalize_factor,
                **kwargs.get("mean_pred_line_kwargs", {}),
            )
            if field_trim[0] is not None and field_trim[1] is not None:
                ax.set_ylim(field_trim[0], field_trim[1])
                ax.set_xlim(coord_trim[0], coord_trim[1])

        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        if kwargs.get("xlabel", None) is not None:
            ax.set_xlabel(kwargs.get("xlabel"))
        if kwargs.get("ylabel", None) is not None:
            ax.set_ylabel(kwargs.get("ylabel"))

        ax.set_title(
            f"Mean Abs. Error: {mae:.2e}. Max Abs. Error: {max_error:.2e}",
            **kwargs.get("title_kwargs", {}),
        )
        ax.set_aspect("equal", adjustable="box")

    else:
        if plot_coord == "x":
            pts = lines.points[:, 0]
        elif plot_coord == "y":
            pts = lines.points[:, 1]
        elif plot_coord == "z":
            pts = lines.points[:, 2]

        true_data = lines.point_data[field_true]
        pred_data = lines.point_data[field_pred]

        if len(true_data.shape) > 1:
            true_data = np.linalg.norm(true_data, axis=1)
            pred_data = np.linalg.norm(pred_data, axis=1)

        sorted_indices = np.argsort(pts)
        pts = pts[sorted_indices]
        true_data = true_data[sorted_indices]
        pred_data = pred_data[sorted_indices]

        max_error = (
            np.max(np.abs((true_data - pred_data)[sorted_indices])) / normalize_factor
        )
        mae = (
            np.mean(np.abs((true_data - pred_data)[sorted_indices])) / normalize_factor
        )

        if coord_trim[0] is not None and coord_trim[1] is not None:
            mask = (pts >= coord_trim[0]) & (pts <= coord_trim[1])
            pts = pts[mask]
            true_data = true_data[mask]
            pred_data = pred_data[mask]

        if flip:
            ax.plot(
                true_data / normalize_factor, pts, **kwargs.get("true_line_kwargs", {})
            )
            ax.plot(
                pred_data / normalize_factor, pts, **kwargs.get("pred_line_kwargs", {})
            )
            if field_trim[0] is not None and field_trim[1] is not None:
                ax.set_xlim(field_trim[0], field_trim[1])
                ax.set_ylim(coord_trim[0], coord_trim[1])
        else:
            ax.plot(
                pts, true_data / normalize_factor, **kwargs.get("true_line_kwargs", {})
            )
            ax.plot(
                pts, pred_data / normalize_factor, **kwargs.get("pred_line_kwargs", {})
            )
            if field_trim[0] is not None and field_trim[1] is not None:
                ax.set_ylim(field_trim[0], field_trim[1])
                ax.set_xlim(coord_trim[0], coord_trim[1])

        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        ax.set_title(
            f"Mean Abs. Error: {mae:.2e}. Max Abs. Error: {max_error:.2e}",
            **kwargs.get("title_kwargs", {}),
        )
        ax.set_aspect("equal", adjustable="box")

        if kwargs.get("xlabel", None) is not None:
            ax.set_xlabel(kwargs.get("xlabel"))
        if kwargs.get("ylabel", None) is not None:
            ax.set_ylabel(kwargs.get("ylabel"))

    return fig


def get_visible_point_indices(
    mesh, camera_direction, camera_focal_point=None, tol=1e-8
):
    """
    Get indices of points that are visible from a given camera direction.

    Parameters:
    -----------
    mesh : pyvista.PolyData
        The mesh to analyze
    camera_direction : str or array-like
        Camera direction. Can be a string like 'xy', 'yz', 'xz' or a 3D vector
    camera_focal_point : array-like, optional
        Camera focal point. If None, uses mesh center
    tol : float, optional
        Tolerance for point matching

    Returns:
    --------
    numpy.ndarray
        Indices of visible points
    """
    if camera_focal_point is None:
        camera_focal_point = mesh.center

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh)

    # Set camera position based on direction
    if isinstance(camera_direction, str):
        if camera_direction == "xy":
            plotter.view_xy()
        elif camera_direction == "yz":
            plotter.view_yz()
        elif camera_direction == "xz":
            plotter.view_xz()
        else:
            plotter.camera_position = camera_direction
    else:
        plotter.view_vector(camera_direction)

    # Reset camera to fit the mesh properly
    plotter.reset_camera()

    render_window = plotter.renderer.GetRenderWindow()
    render_window.Render()

    select_visible_points = vtk.vtkSelectVisiblePoints()
    select_visible_points.SetInputData(mesh)
    select_visible_points.SetRenderer(plotter.renderer)
    select_visible_points.Update()

    visible = pv.wrap(select_visible_points.GetOutput())
    plotter.close()

    if len(visible.points) == 0:
        return np.array([], dtype=int)

    tree = cKDTree(mesh.points)
    dists, idxs = tree.query(visible.points, distance_upper_bound=tol)
    # Filter out points that didn't match (distance == inf)
    idxs = idxs[np.isfinite(dists)]

    return idxs


def plot_projections_hexbin(meshes, field, direction, grid_size=50):
    """Helper function to plot HexBin plots across multiple meshes"""

    # Aggregate results
    x = []
    y = []
    field_arr = []

    for mesh in meshes:
        x.append(mesh.points[:, 0])
        y.append(mesh.points[:, 1])
        field_arr.append(mesh.point_data[field])

    x = np.concatenate(x)
    y = np.concatenate(y)
    field_arr = np.concatenate(field_arr)

    # Hexbin plot
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    ax = axs[0]
    hb = ax.hexbin(
        x, y, C=field_arr, gridsize=grid_size, reduce_C_function=np.mean, cmap="jet"
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Mean")
    ax.set_aspect("equal", adjustable="datalim")

    if direction in ["-YZ", "-ZX"]:
        ax.invert_yaxis()
    elif direction in ["-XY"]:
        ax.invert_xaxis()

    ax = axs[1]
    hb = ax.hexbin(
        x, y, C=field_arr, gridsize=grid_size, reduce_C_function=np.std, cmap="jet"
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("StdDev")
    ax.set_aspect("equal", adjustable="datalim")

    if direction in ["-YZ", "-ZX"]:
        ax.invert_yaxis()
    elif direction in ["-XY"]:
        ax.invert_xaxis()

    plt.tight_layout()

    return fig
