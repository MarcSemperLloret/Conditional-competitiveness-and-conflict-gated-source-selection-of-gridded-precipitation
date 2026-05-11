#!/usr/bin/env python
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pandas as pd

import build_delta_mae_by_support_score_and_regime_v1 as base


@contextmanager
def temporary_time_window(start: pd.Timestamp, end: pd.Timestamp):
    old_start, old_end = base.TIME_START, base.TIME_END
    base.TIME_START = start
    base.TIME_END = end
    try:
        yield
    finally:
        base.TIME_START = old_start
        base.TIME_END = old_end


def year_window(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(f"{year}-01-01 00:00:00"), pd.Timestamp(f"{year + 1}-01-01 00:00:00")


def build_year_prediction_frame(year: int) -> pd.DataFrame:
    start, end = year_window(year)
    with temporary_time_window(start, end):
        return base.build_prediction_frame()


def add_leavecell_support_metrics(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    active_radius_km: float = 15.0,
    max_neighbors: int = 64,
) -> pd.DataFrame:
    out = df.copy()

    static_df = pd.read_parquet(base.STATIC_PARQUET).copy()
    static_df = static_df.sort_values(["imerg_local_lon_idx", "imerg_local_lat_idx"]).reset_index(drop=True)
    static_df["cell_idx"] = np.arange(static_df.shape[0], dtype=np.int32)
    grid_to_idx = dict(zip(static_df["grid_id"], static_df["cell_idx"], strict=False))

    time_axis = pd.date_range(start, end - pd.Timedelta(hours=1), freq="1h")
    time_to_idx = {timestamp: idx for idx, timestamp in enumerate(time_axis)}
    out["cell_idx"] = out["grid_id"].map(grid_to_idx).astype(np.int32)
    out["time_idx"] = out["time_utc"].map(time_to_idx).astype(np.int32)

    station_df = base.load_station_inventory()
    with temporary_time_window(start, end):
        obs_matrix = base.load_station_obs_matrix(time_axis, station_df)
    neighbor_idx, neighbor_dist = base.build_neighbor_rank_tables(static_df, station_df, max_neighbors=max_neighbors)

    cell_idx = out["cell_idx"].to_numpy(dtype=np.int32, copy=False)
    time_idx = out["time_idx"].to_numpy(dtype=np.int32, copy=False)
    row_neighbor_idx = neighbor_idx[cell_idx]
    row_neighbor_dist = neighbor_dist[cell_idx]
    safe_neighbor_idx = np.where(row_neighbor_idx >= 0, row_neighbor_idx, 0)
    values = obs_matrix[time_idx[:, None], safe_neighbor_idx]

    valid_neighbor = row_neighbor_idx >= 0
    active_neighbor = valid_neighbor & np.isfinite(values) & np.isfinite(row_neighbor_dist)
    clipped_dist = np.clip(row_neighbor_dist.astype(np.float32, copy=False), 0.1, None)
    weights = np.where(active_neighbor, 1.0 / np.square(clipped_dist), 0.0).astype(np.float32, copy=False)
    weight_sum = weights.sum(axis=1, dtype=np.float32)
    weight_sq_sum = np.square(weights, dtype=np.float32).sum(axis=1, dtype=np.float32)

    nearest_active_km = np.where(active_neighbor, row_neighbor_dist, np.inf).min(axis=1)
    nearest_active_km = np.where(np.isfinite(nearest_active_km), nearest_active_km, np.nan).astype(np.float32)

    out["leavecell_active_15km"] = (
        active_neighbor & np.isfinite(row_neighbor_dist) & (row_neighbor_dist <= active_radius_km)
    ).sum(axis=1, dtype=np.int16)
    out["leavecell_nearest_active_km"] = nearest_active_km
    out["leavecell_idw_weight_sum"] = weight_sum
    neff = np.full(weight_sum.shape, np.nan, dtype=np.float32)
    np.divide(
        np.square(weight_sum, dtype=np.float32),
        weight_sq_sum,
        out=neff,
        where=weight_sq_sum > 0.0,
    )
    out["leavecell_idw_neff"] = neff
    out["gridded_mean_mm"] = out[list(base.EO_COLUMNS.values())].mean(axis=1)

    return out


def build_year_prediction_frame_with_support(year: int) -> pd.DataFrame:
    start, end = year_window(year)
    df = build_year_prediction_frame(year)
    return add_leavecell_support_metrics(df, start=start, end=end)

