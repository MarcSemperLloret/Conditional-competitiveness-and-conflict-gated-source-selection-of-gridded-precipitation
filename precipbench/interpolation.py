"""Leakage-controlled leave-cell IDW interpolation."""
from __future__ import annotations

import numpy as np


def haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Great-circle distance in km between two sets of lat/lon points.

    Parameters
    ----------
    lat1, lon1 : (N,) arrays  — source points (cells)
    lat2, lon2 : (M,) arrays  — destination points (stations)

    Returns
    -------
    (N, M) distance matrix in km
    """
    lat1_r = np.deg2rad(lat1)[:, None]
    lon1_r = np.deg2rad(lon1)[:, None]
    lat2_r = np.deg2rad(lat2)[None, :]
    lon2_r = np.deg2rad(lon2)[None, :]
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))
    return 6371.0 * c


def build_neighbor_table(
    cell_lats: np.ndarray,
    cell_lons: np.ndarray,
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    station_cell_ids: np.ndarray,
    cell_ids: np.ndarray,
    k: int = 64,
    radius_km: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute leave-cell neighbor indices and distances for IDW.

    Same-cell stations are excluded (leave-cell constraint).

    Parameters
    ----------
    cell_lats, cell_lons : (C,) arrays of grid cell centroids
    station_lats, station_lons : (S,) arrays of station coordinates
    station_cell_ids : (S,) cell ID for each station (used for exclusion)
    cell_ids : (C,) cell ID for each grid cell
    k : number of nearest neighbors to retain
    radius_km : optional distance cap; neighbors beyond this are dropped

    Returns
    -------
    neighbor_idx : (C, k) int32 — station index, -1 if unused
    neighbor_dist : (C, k) float32 — distance in km, NaN if unused
    """
    dist = haversine_km(cell_lats, cell_lons, station_lats, station_lons)

    # Leave-cell exclusion: set same-cell distance to inf
    same_cell = cell_ids[:, None] == station_cell_ids[None, :]
    dist[same_cell] = np.inf

    n_cells = dist.shape[0]
    neighbor_idx = np.full((n_cells, k), -1, dtype=np.int32)
    neighbor_dist = np.full((n_cells, k), np.nan, dtype=np.float32)

    for i in range(n_cells):
        order = np.argsort(dist[i])[:k]
        finite = np.isfinite(dist[i, order])
        keep = order[finite]
        if radius_km is not None:
            keep = keep[dist[i, keep] <= radius_km]
        n = keep.size
        neighbor_idx[i, :n] = keep
        neighbor_dist[i, :n] = dist[i, keep].astype(np.float32)

    return neighbor_idx, neighbor_dist


def leavecell_idw(
    obs_matrix: np.ndarray,
    neighbor_idx: np.ndarray,
    neighbor_dist: np.ndarray,
    power: float = 2.0,
) -> np.ndarray:
    """Compute leave-cell IDW predictions for all cell-hours.

    Parameters
    ----------
    obs_matrix : (T, S) float32 — hourly station observations, NaN if missing
    neighbor_idx : (C, k) int32 — precomputed neighbor indices from
        :func:`build_neighbor_table`
    neighbor_dist : (C, k) float32 — precomputed neighbor distances in km
    power : IDW distance-decay exponent (default 2)

    Returns
    -------
    (T, C) float32 prediction matrix, NaN where no neighbor is available
    """
    n_hours, _ = obs_matrix.shape
    n_cells = neighbor_idx.shape[0]
    out = np.full((n_hours, n_cells), np.nan, dtype=np.float32)

    for ci in range(n_cells):
        idx = neighbor_idx[ci]
        valid = idx >= 0
        idx = idx[valid]
        if idx.size == 0:
            continue
        dist = np.clip(neighbor_dist[ci, valid].astype(np.float64), 0.1, None)
        weights = (1.0 / dist**power).astype(np.float32)
        vals = obs_matrix[:, idx]          # (T, k_valid)
        active = np.isfinite(vals)
        denom = (active * weights).sum(axis=1)
        numer = (np.where(active, vals, 0.0) * weights).sum(axis=1)
        pred = np.full(n_hours, np.nan, dtype=np.float32)
        mask = denom > 0
        pred[mask] = (numer[mask] / denom[mask]).astype(np.float32)
        out[:, ci] = np.clip(pred, 0.0, None)

    return out
