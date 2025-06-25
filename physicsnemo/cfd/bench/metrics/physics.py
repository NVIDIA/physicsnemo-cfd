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

import vtk
import pyvista as pv
import numpy as np
from tqdm import tqdm
from numba import njit, prange, cuda, float64


@njit
def edges_to_adjacency(
    sorted_bidirectional_edges: np.ndarray, n_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a sorted bidirectional edge list to an adjacency list.

    Parameters
    ----------
    sorted_bidirectional_edges : np.ndarray
        A 2D array of shape (n_edges, 2) where each row contains the start
        and end indices of an edge. Edges are sorted by increasing start index,
        then increasing end index. Each edge is listed twice, once in each direction.

    n_points : int
        The number of points in the mesh. Usually equivalent to
        np.max(sorted_bidirectional_edges) + 1, though only true if there are
        no unconnected points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of two numpy arrays. The first array is a 1D array of shape (n_points,)
        containing the indices of the neighbors of each point. The second array is a 1D
        array of shape (n_points,) containing the indices of the neighbors of each point.

    Examples
    --------
    >>> sorted_bidirectional_edges = np.array([
    ...     [0, 1],
    ...     [0, 2],
    ...     [0, 3],
    ...     [1, 0],
    ...     [1, 2],
    ...     [1, 15],
    ...     [1, 16],
    ...     [2, 0],
    ...     [2, 1],
    ...     [2, 17],
    ...     ...
    ... ])
    >>> adjacency = edges_to_adjacency(sorted_bidirectional_edges, 18)
    """
    n_edges = len(sorted_bidirectional_edges)
    offsets = np.zeros(n_points + 1, dtype=sorted_bidirectional_edges.dtype)
    indices = np.zeros(n_edges, dtype=sorted_bidirectional_edges.dtype)

    edge_idx = 0
    for adj_index in range(n_points):
        start_offset = offsets[adj_index]
        while edge_idx < n_edges:
            start_idx = sorted_bidirectional_edges[edge_idx, 0]
            if start_idx == adj_index:
                indices[start_offset] = sorted_bidirectional_edges[edge_idx, 1]
                start_offset += 1
            elif start_idx > adj_index:
                break
            edge_idx += 1
        offsets[adj_index + 1] = start_offset

    return offsets, indices


edges_to_adjacency(np.zeros((0, 2), dtype=np.int64), 0)  # Does the precompilation


def unique_axis0(array):
    """
    A faster version of np.unique(array, axis=0) for 2D arrays.

    Returns
    -------
    np.ndarray
        The unique rows of the input array.

    Notes
    -----
    ~25x faster than np.unique(array, axis=0) on PyVista brain mesh non-unique edge array.
    """
    idxs = np.lexsort(array.T[::-1])
    array = array[idxs]
    unique_idxs = np.empty(len(array), dtype=np.bool_)
    unique_idxs[0] = True
    unique_idxs[1:] = np.any(array[:-1, :] != array[1:, :], axis=-1)
    return array[unique_idxs]


def get_edges(mesh: pv.DataSet) -> np.ndarray:
    """
    Given a mesh, returns a 2D array of shape (n_edges, 2) where each row contains the start
    and end indices of an edge. Edges are sorted by increasing start index, then increasing end index.
    Each edge is listed twice, once in each direction.

    Parameters
    ----------
    mesh : pv.DataSet
        The input mesh.

    Returns
    -------
    np.ndarray
        A 2D array of shape (n_edges, 2) with the start and end indices of each edge.

    Examples
    --------
    These should be identical:

    >>> edges = get_edges(mesh)
    >>> order = np.lexsort(edges.T[::-1])
    >>> edges = edges[order]

    and

    >>> edges = mesh.extract_all_edges(use_all_points=True, clear_data=True).lines.reshape(-1, 3)[:, 1:]
    """
    edges_from_all_cell_types: list[np.ndarray] = []

    cells_dict = mesh.cells_dict

    for cell_type, cells in cells_dict.items():
        ### Determine the canonical edges for this particular cell type
        # First, create a canonical cell (i.e., a mesh with a single cell of this same type)
        # The purpose is to dynamically determine edge connectivity for this cell type, which
        # we will then vectorize onto all cells of this type in the mesh
        n_vertices_per_cell = cells.shape[1]
        canonical_cell = pv.UnstructuredGrid(
            np.concatenate(
                [np.array([n_vertices_per_cell]), np.arange(n_vertices_per_cell)]
            ),
            [cell_type],
            np.zeros((n_vertices_per_cell, 3), dtype=float),
        )
        canonical_edges = canonical_cell.extract_all_edges(
            use_all_points=True, clear_data=True
        ).lines.reshape(-1, 3)[:, 1:]

        ### Now, map this onto all cells of this type in the mesh
        edges_from_this_cell_type = np.empty(
            (len(canonical_edges) * len(cells), 2), dtype=np.int64
        )
        for i, edge in enumerate(canonical_edges):
            edges_from_this_cell_type[i * len(cells) : (i + 1) * len(cells)] = cells[
                :, edge
            ]

        edges_from_all_cell_types.append(edges_from_this_cell_type)

    if len(edges_from_all_cell_types) == 1:
        # No need to make a memory copy in this case (which np.concatenate forces)
        edges = edges_from_all_cell_types[0]

    else:
        edges = np.concatenate(edges_from_all_cell_types, axis=0)

    ### Now, eliminate duplicate edges
    # Identical to np.sort(edges, axis=1) for Nx2 arrays, but faster
    edges = np.where(np.diff(edges, axis=1) >= 0, edges, edges[:, ::-1])

    edges = unique_axis0(edges)

    return edges


def build_point_adjacency(
    mesh: pv.DataSet, progress_bar=False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an adjacency list for the points in a mesh.

    Parameters
    ----------
    mesh : pv.DataSet
        The input mesh.

    progress_bar : bool, optional
        Whether to display a progress bar during computation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of two numpy arrays. The first array is a 1D array of shape (n_points,)
        containing the indices of the neighbors of each point. The second array is a 1D
        array of shape (n_points,) containing the indices of the neighbors of each point.
    """
    # edges = get_edges(mesh)

    ### This is the old way of getting edges (slower) + consistency check
    edge_mesh = mesh.extract_all_edges(use_all_points=True, clear_data=True)
    edges = edge_mesh.lines.reshape(-1, 3)[:, 1:]
    # order = np.lexsort(edges.T[::-1])
    # edges = edges[order]
    # assert np.all(edges == edges_old)

    # Includes not only edge [a, b] but also edge [b, a]
    bidirectional_edges = np.concatenate((edges, edges[:, ::-1]), axis=0)

    # Puts edges in order of increasing start point index, then increasing end point index
    order = np.lexsort((bidirectional_edges[:, 1], bidirectional_edges[:, 0]))
    sorted_bidirectional_edges = bidirectional_edges[order]

    adjacency = edges_to_adjacency(sorted_bidirectional_edges, mesh.n_points)

    return adjacency


@njit(fastmath=True, parallel=False)
def compute_gradients(points, u, adjacency) -> np.ndarray:
    """
    Compute the gradients of a vector field at each point in a mesh.

    Parameters
    ----------
    points : np.ndarray
        The points of the mesh.

    u : np.ndarray
        The vector field.

    adjacency : tuple[np.ndarray, np.ndarray]
        The adjacency list.

    Returns
    -------
    np.ndarray
        The gradients of the vector field. Shape is (N, D, 3), where:
        - N is the number of points,
        - D is the number of vector components (e.g., 3 for velocity), and
        - 3 is the number of spatial dimensions.
    """
    assert u.dtype == points.dtype, "u and points must have the same dtype"
    dtype = u.dtype
    N, D = u.shape
    grad_u = np.empty((N, D, 3), dtype=dtype)  # (N points, D fields, 3 spatial grads)

    offsets, indices = adjacency
    for i in prange(N):
        start = offsets[i]
        end = offsets[i + 1]
        neighbors = indices[start:end]

        dv = points[neighbors] - points[i]  # (M, 3)
        du = u[neighbors] - u[i]  # (M, D)

        w_squared = 1.0 / ((dv**2).sum(axis=1) + 1e-8)  # (M,)
        W = np.diag(w_squared).astype(dtype)  # (M, M)

        A = dv.T @ W @ dv  # (3, 3)
        B = dv.T @ W @ du  # (3, D)

        grad, _, _, _ = np.linalg.lstsq(A, B, rcond=-1)  # (3, D)
        grad_u[i] = grad.T  # Store as (D, 3)

    return grad_u


MAX_NEIGHBORS = 128
MAX_CHANNELS = 8


@cuda.jit
def compute_gradients_gpu(points, u, grad_u, offsets, indices, N, D):
    """
    Compute the gradients of a field u at points using a GPU.
    """
    i = cuda.grid(1)
    if i >= N:
        return

    start = offsets[i]
    end = offsets[i + 1]
    num_neighbors = end - start

    if num_neighbors < 3 or num_neighbors > MAX_NEIGHBORS:
        for d in range(D):
            for j in range(3):
                grad_u[i, d, j] = 0.0
        return

    dv = cuda.local.array((MAX_NEIGHBORS, 3), dtype=float64)
    du = cuda.local.array((MAX_NEIGHBORS, MAX_CHANNELS), dtype=float64)
    w = cuda.local.array(MAX_NEIGHBORS, dtype=float64)
    A = cuda.local.array((3, 3), dtype=float64)
    B = cuda.local.array((3, MAX_CHANNELS), dtype=float64)

    for n in range(num_neighbors):
        nid = indices[start + n]
        for j in range(3):
            dv[n, j] = points[nid, j] - points[i, j]
        for d in range(D):
            du[n, d] = u[nid, d] - u[i, d]
        w[n] = 1.0 / (dv[n, 0] ** 2 + dv[n, 1] ** 2 + dv[n, 2] ** 2 + 1e-8)

    for j in range(3):
        for k in range(3):
            A[j, k] = 0.0
        for d in range(D):
            B[j, d] = 0.0

    # Accumulate A and B
    for n in range(num_neighbors):
        for j in range(3):
            for k in range(3):
                A[j, k] += dv[n, j] * w[n] * dv[n, k]
            for d in range(D):
                B[j, d] += dv[n, j] * w[n] * du[n, d]

    # solve using explicit inverse
    det = (
        A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
        - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
        + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
    )

    if abs(det) < 1e-8:
        return

    invA = cuda.local.array((3, 3), dtype=float64)
    invA[0, 0] = (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) / det
    invA[0, 1] = (A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]) / det
    invA[0, 2] = (A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]) / det
    invA[1, 0] = (A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]) / det
    invA[1, 1] = (A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]) / det
    invA[1, 2] = (A[0, 2] * A[1, 0] - A[0, 0] * A[1, 2]) / det
    invA[2, 0] = (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]) / det
    invA[2, 1] = (A[0, 1] * A[2, 0] - A[0, 0] * A[2, 1]) / det
    invA[2, 2] = (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]) / det

    for d in range(D):
        for j in range(3):
            grad_u[i, d, j] = 0.0
            for k in range(3):
                grad_u[i, d, j] += invA[j, k] * B[k, d]


def _compute_continuity(points, u, adjacency, device="cpu"):
    if device == "cpu":
        grad_u = compute_gradients(points, u, adjacency)
    elif device == "gpu":
        import cupy as cp

        offsets, indices = adjacency
        points = cp.asarray(points)
        u = cp.asarray(u)
        offsets = cp.asarray(offsets)
        indices = cp.asarray(indices)
        grad_u = cp.zeros((u.shape[0], u.shape[1], 3), dtype=u.dtype)
        threads_per_block = 256
        blocks_per_grid = (points.shape[0] + threads_per_block - 1) // threads_per_block
        compute_gradients_gpu[blocks_per_grid, threads_per_block](
            points, u, grad_u, offsets, indices, points.shape[0], u.shape[1]
        )
        grad_u = cp.asnumpy(grad_u)
    return sum(
        [grad_u[:, i, i] for i in range(3)]
    )  # Trace of the velocity gradient tensor


def _compute_momentum(points, u, p, mu, rho, adjacency, device="cpu"):
    p = p[:, np.newaxis]
    mu = mu[:, np.newaxis]
    u_p = np.concatenate((u, p), axis=1)

    if device == "cpu":
        grad_u_p = compute_gradients(points, u_p, adjacency)
    elif device == "gpu":
        import cupy as cp

        offsets, indices = adjacency
        points = cp.asarray(points)
        u_p = cp.asarray(u_p)
        offsets = cp.asarray(offsets)
        indices = cp.asarray(indices)
        grad_u_p = cp.zeros((u_p.shape[0], u_p.shape[1], 3), dtype=u_p.dtype)
        threads_per_block = 256
        blocks_per_grid = (points.shape[0] + threads_per_block - 1) // threads_per_block
        compute_gradients_gpu[blocks_per_grid, threads_per_block](
            points, u_p, grad_u_p, offsets, indices, points.shape[0], u_p.shape[1]
        )
        grad_u_p = cp.asnumpy(grad_u_p)

    grad_u = grad_u_p[:, 0:3, :]
    grad_p = grad_u_p[:, 3:4, :]

    tau_xx = 2 * mu * grad_u[:, 0:1, 0]
    tau_yy = 2 * mu * grad_u[:, 1:2, 1]
    tau_zz = 2 * mu * grad_u[:, 2:3, 2]
    tau_xy = mu * (grad_u[:, 0:1, 1] + grad_u[:, 1:2, 0])
    tau_xz = mu * (grad_u[:, 0:1, 2] + grad_u[:, 2:3, 0])
    tau_yz = mu * (grad_u[:, 1:2, 2] + grad_u[:, 2:3, 1])
    tau = np.concatenate((tau_xx, tau_yy, tau_zz, tau_xy, tau_xz, tau_yz), axis=1)

    if device == "cpu":
        tau_grad = compute_gradients(points, tau, adjacency)
    elif device == "gpu":
        import cupy as cp

        offsets, indices = adjacency
        points = cp.asarray(points)
        tau = cp.asarray(tau)
        offsets = cp.asarray(offsets)
        indices = cp.asarray(indices)
        tau_grad = cp.zeros((tau.shape[0], tau.shape[1], 3), dtype=tau.dtype)
        threads_per_block = 256
        blocks_per_grid = (points.shape[0] + threads_per_block - 1) // threads_per_block
        compute_gradients_gpu[blocks_per_grid, threads_per_block](
            points, tau, tau_grad, offsets, indices, points.shape[0], tau.shape[1]
        )
        tau_grad = cp.asnumpy(tau_grad)

    tau_xx_grad = tau_grad[:, 0:1, :]
    tau_yy_grad = tau_grad[:, 1:2, :]
    tau_zz_grad = tau_grad[:, 2:3, :]
    tau_xy_grad = tau_grad[:, 3:4, :]
    tau_xz_grad = tau_grad[:, 4:5, :]
    tau_yz_grad = tau_grad[:, 5:6, :]

    momentum_x = (
        rho
        * (
            u[:, 0:1] * grad_u[:, 0:1, 0]
            + u[:, 1:2] * grad_u[:, 0:1, 1]
            + u[:, 2:3] * grad_u[:, 0:1, 2]
        )
        + grad_p[:, :, 0]
        - tau_xx_grad[:, :, 0]
        - tau_xy_grad[:, :, 1]
        - tau_xz_grad[:, :, 2]
    )
    momentum_y = (
        rho
        * (
            u[:, 0:1] * grad_u[:, 1:2, 0]
            + u[:, 1:2] * grad_u[:, 1:2, 1]
            + u[:, 2:3] * grad_u[:, 1:2, 2]
        )
        + grad_p[:, :, 1]
        - tau_xy_grad[:, :, 0]
        - tau_yy_grad[:, :, 1]
        - tau_yz_grad[:, :, 2]
    )
    momentum_z = (
        rho
        * (
            u[:, 0:1] * grad_u[:, 2:3, 0]
            + u[:, 1:2] * grad_u[:, 2:3, 1]
            + u[:, 2:3] * grad_u[:, 2:3, 2]
        )
        + grad_p[:, :, 2]
        - tau_xz_grad[:, :, 0]
        - tau_yz_grad[:, :, 1]
        - tau_zz_grad[:, :, 2]
    )

    return np.concatenate((momentum_x, momentum_y, momentum_z), axis=1)


def compute_continuity_residuals(
    mesh, true_velocity_field="U", predicted_velocity_field="U_pred", device="cpu"
):
    """
    Compute the continuity residuals of a velocity field.

    Parameters
    ----------
    mesh : pv.DataSet
        The input mesh.

    true_velocity_field : str, optional
        The name of the true velocity field.

    predicted_velocity_field : str, optional
        The name of the predicted velocity field.

    device : str, optional
        The device to use for the computation.

    Returns
    -------
    pv.DataSet
        The input mesh with the continuity residuals added to the point data.
    """
    adjacency = build_point_adjacency(mesh)

    # Cast fields to float64 for accurate gradient computation
    assert (
        true_velocity_field in mesh.point_data.keys()
    ), f"{true_velocity_field} not found in point mesh."
    assert (
        predicted_velocity_field in mesh.point_data.keys()
    ), f"{predicted_velocity_field} not found in point mesh."

    u_true = mesh.point_data[true_velocity_field].astype(np.float64)
    u_pred = mesh.point_data[predicted_velocity_field].astype(np.float64)

    cont_true = _compute_continuity(
        mesh.points.astype(np.float64), u_true.astype(np.float64), adjacency, device
    )
    cont_pred = _compute_continuity(
        mesh.points.astype(np.float64), u_pred.astype(np.float64), adjacency, device
    )

    mesh.point_data["Continuity"] = cont_true
    mesh.point_data["ContinuityPred"] = cont_pred

    return mesh


def compute_momentum_residuals(
    mesh,
    true_velocity_field="U",
    predicted_velocity_field="U_pred",
    true_pressure_field="p",
    predicted_pressure_field="p_pred",
    true_nu_field="nut",
    predicted_nu_field="nut_pred",
    nu=1.5e-5,
    rho=1.0,
    device="cpu",
):
    """
    Compute the momentum residuals of a velocity field using the RANS formulation.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Reynolds-averaged_Navier%E2%80%93Stokes_equations

    Parameters
    ----------
    mesh : pv.DataSet
        The input mesh.

    true_velocity_field : str, optional
        The name of the true velocity field.

    predicted_velocity_field : str, optional
        The name of the predicted velocity field.

    true_pressure_field : str, optional
        The name of the true pressure field.

    predicted_pressure_field : str, optional
        The name of the predicted pressure field.

    true_nu_field : str, optional
        The name of the true nu field.

    predicted_nu_field : str, optional
        The name of the predicted nu field.

    nu : float, optional
        The kinematic viscosity.

    rho : float, optional
        The density.

    device : str, optional
        The device to use for the computation.

    Returns
    -------
    pv.DataSet
        The input mesh with the momentum residuals added to the point data.
    """
    adjacency = build_point_adjacency(mesh)

    # Cast fields to float64 for accurate gradient computation
    assert (
        true_velocity_field in mesh.point_data.keys()
    ), f"{true_velocity_field} not found in point mesh."
    assert (
        predicted_velocity_field in mesh.point_data.keys()
    ), f"{predicted_velocity_field} not found in point mesh."

    assert (
        true_pressure_field in mesh.point_data.keys()
    ), f"{true_pressure_field} not found in point mesh."
    assert (
        predicted_pressure_field in mesh.point_data.keys()
    ), f"{predicted_pressure_field} not found in point mesh."

    assert (
        true_nu_field in mesh.point_data.keys()
    ), f"{true_nu_field} not found in point mesh."
    assert (
        predicted_nu_field in mesh.point_data.keys()
    ), f"{predicted_nu_field} not found in point mesh."

    u_true = mesh.point_data[true_velocity_field].astype(np.float64)
    u_pred = mesh.point_data[predicted_velocity_field].astype(np.float64)

    p_true = mesh.point_data[true_pressure_field].astype(np.float64)
    p_pred = mesh.point_data[predicted_pressure_field].astype(np.float64)

    nu_true = mesh.point_data[true_nu_field].astype(np.float64)
    nu_pred = mesh.point_data[predicted_nu_field].astype(np.float64)

    mu_true = rho * (nu + nu_true)
    mu_pred = rho * (nu + nu_pred)

    momentum_true = _compute_momentum(
        mesh.points.astype(np.float64), u_true, p_true, mu_true, rho, adjacency, device
    )
    momentum_pred = _compute_momentum(
        mesh.points.astype(np.float64), u_pred, p_pred, mu_pred, rho, adjacency, device
    )

    mesh.point_data["Momentum"] = momentum_true
    mesh.point_data["MomentumPred"] = momentum_pred

    return mesh


if __name__ == "__main__":
    import pyvista as pv
    from pyvista import examples

    mesh = examples.download_electronics_cooling()["air"]
    # mesh = mesh.cast_to_unstructured_grid()
    # mesh = pv.read(r"/raid/psharpe/automotive_cfd_data_generation/reference_openfoam_configs/volume_4_predicted_new.vtu")

    adj = build_point_adjacency(mesh)
    mesh = compute_continuity_residuals(mesh, "U", "U", device="gpu")
