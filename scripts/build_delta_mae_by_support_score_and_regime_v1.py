#!/usr/bin/env python
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESTRICTED_DATA_DIR = Path(os.environ.get("BENCHMARK_RESTRICTED_DATA_DIR", ROOT / "data" / "restricted"))
CANONICAL_PARQUET = RESTRICTED_DATA_DIR / "canonical_cell_hour_imerg_v1.parquet"
STATIC_PARQUET = RESTRICTED_DATA_DIR / "imerg_cell_static_v1.parquet"
STATION_CSV = RESTRICTED_DATA_DIR / "avamet_station_inventory_cv_imerg.csv"
AVAMET_HOURLY = RESTRICTED_DATA_DIR / "avamet_cv_hourly_2019_2023.parquet"
OUTPUT_DIR = ROOT / "results"

TIME_START = pd.Timestamp("2023-01-01 00:00:00")
TIME_END = pd.Timestamp("2024-01-01 00:00:00")
IDW_POWER = 2.0
LOCAL_BASELINES = [
    "domain_idw_leavecell_knn08",
    "domain_idw_leavecell_knn64",
    "domain_idw_leavecell_knn64_r15km",
]
FIXED_LOCAL_BASELINE = "domain_idw_leavecell_knn64"
EO_COLUMNS = {
    "source_imerg": "imerg_mm",
    "source_era5": "era5_mm",
    "source_euradclim": "euradclim_on_imerg_mm",
}
SCENARIOS = {
    "wide_ge_1": 1,
    "robust_ge_2": 2,
}
KEEP_REGIME_STRATIFIERS = ["overall", "observable_conflict", "target_intensity"]

SUPPORT_BIN_ORDER = {
    "all": 0,
    "single_station_like_le_0.34": 1,
    "two_station_like_0.34_0.67": 2,
    "three_plus_like_gt_0.67": 3,
}
REGIME_STRATIFIER_ORDER = {
    "overall": 0,
    "observable_conflict": 1,
    "target_intensity": 2,
}
REGIME_STRATUM_ORDER = {
    "overall": {"all": 0},
    "observable_conflict": {"low_le_p75": 0, "mid_p75_p95": 1, "tail_gt_p95": 2},
    "target_intensity": {
        "=0": 0,
        "(0,0.1]": 1,
        "(0.1,1]": 2,
        "(1,5]": 3,
        "(5,10]": 4,
        "(10,20]": 5,
        ">20": 6,
    },
}


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1_rad = np.deg2rad(lat1)[:, None]
    lon1_rad = np.deg2rad(lon1)[:, None]
    lat2_rad = np.deg2rad(lat2)[None, :]
    lon2_rad = np.deg2rad(lon2)[None, :]
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))
    return 6371.0 * c


def build_fine_target_bins(series: pd.Series) -> pd.Series:
    return pd.cut(
        series,
        bins=[-1e-9, 0.0, 0.1, 1.0, 5.0, 10.0, 20.0, np.inf],
        labels=["=0", "(0,0.1]", "(0.1,1]", "(1,5]", "(5,10]", "(10,20]", ">20"],
        ordered=True,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | None]:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(finite):
        return {
            "n_eval": 0,
            "mae_mm": None,
            "rmse_mm": None,
            "corr": None,
            "mae_positive_target_mm": None,
            "coverage_share": 0.0,
        }

    y_t = y_true[finite].astype(np.float64, copy=False)
    y_p = y_pred[finite].astype(np.float64, copy=False)
    error = y_p - y_t
    out: dict[str, float | int | None] = {
        "n_eval": int(y_t.size),
        "mae_mm": float(np.mean(np.abs(error))),
        "rmse_mm": float(np.sqrt(np.mean(np.square(error)))),
        "corr": None,
        "mae_positive_target_mm": None,
        "coverage_share": float(y_t.size) / float(y_true.size) if y_true.size else 0.0,
    }
    if y_t.size >= 2 and np.std(y_t) > 0 and np.std(y_p) > 0:
        out["corr"] = float(np.corrcoef(y_t, y_p)[0, 1])
    pos_mask = y_t > 0.0
    if np.any(pos_mask):
        out["mae_positive_target_mm"] = float(np.mean(np.abs(error[pos_mask])))
    return out


def load_common_overlap_2023() -> pd.DataFrame:
    con = duckdb.connect()
    query = f"""
        SELECT
            CAST(time_utc AS TIMESTAMP) AS time_utc,
            grid_id,
            imerg_mm,
            era5_mm,
            euradclim_on_imerg_mm,
            pairwise_disagreement_imerg_era5_abs_mm,
            pairwise_disagreement_imerg_euradclim_abs_mm,
            pairwise_disagreement_era5_euradclim_abs_mm,
            elevation_m,
            distance_to_coast_km,
            avamet_agg_mm,
            avamet_n_active,
            avamet_support_score_v0
        FROM read_parquet('{CANONICAL_PARQUET}')
        WHERE CAST(time_utc AS TIMESTAMP) >= ?
          AND CAST(time_utc AS TIMESTAMP) < ?
          AND avamet_n_active >= 1
          AND isfinite(avamet_agg_mm)
          AND isfinite(imerg_mm)
          AND isfinite(era5_mm)
          AND isfinite(euradclim_on_imerg_mm)
        ORDER BY time_utc, grid_id
    """
    df = con.execute(query, [TIME_START, TIME_END]).fetchdf()
    con.close()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=False)
    return df


def add_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    pair_cols = [
        "pairwise_disagreement_imerg_era5_abs_mm",
        "pairwise_disagreement_imerg_euradclim_abs_mm",
        "pairwise_disagreement_era5_euradclim_abs_mm",
    ]
    pair_stack = np.stack([df[col].to_numpy(dtype=np.float32, copy=False) for col in pair_cols], axis=1)
    valid = np.isfinite(pair_stack)
    denom = valid.sum(axis=1).astype(np.float32, copy=False)
    safe_sum = np.where(valid, pair_stack, 0.0).sum(axis=1, dtype=np.float32)
    observable_conflict = np.where(denom > 0, safe_sum / denom, np.nan).astype(np.float32)
    df["observable_conflict_nonref_v0"] = observable_conflict
    q75 = float(df["observable_conflict_nonref_v0"].quantile(0.75))
    q95 = float(df["observable_conflict_nonref_v0"].quantile(0.95))
    df["observable_conflict_bin"] = np.select(
        [
            df["observable_conflict_nonref_v0"] <= q75,
            (df["observable_conflict_nonref_v0"] > q75) & (df["observable_conflict_nonref_v0"] <= q95),
            df["observable_conflict_nonref_v0"] > q95,
        ],
        ["low_le_p75", "mid_p75_p95", "tail_gt_p95"],
        default="unknown",
    )
    df["support_regime_label"] = np.where(df["avamet_n_active"] >= 2, "robust_ge_2", "single_station")
    df["target_intensity_bin_v2"] = build_fine_target_bins(df["avamet_agg_mm"]).astype(str)
    df["support_score_bin"] = np.select(
        [
            df["avamet_support_score_v0"] <= 0.34,
            (df["avamet_support_score_v0"] > 0.34) & (df["avamet_support_score_v0"] <= 0.67),
            df["avamet_support_score_v0"] > 0.67,
        ],
        [
            "single_station_like_le_0.34",
            "two_station_like_0.34_0.67",
            "three_plus_like_gt_0.67",
        ],
        default="unknown",
    )
    return df


def load_station_inventory() -> pd.DataFrame:
    station_df = pd.read_csv(STATION_CSV)
    station_df = station_df.loc[station_df["imerg_cell_in_cv_polygon"].fillna(False).astype(bool)].copy()
    station_df["grid_id"] = (
        "imerg_"
        + station_df["imerg_cv_lon_idx"].astype(int).astype(str)
        + "_"
        + station_df["imerg_cv_lat_idx"].astype(int).astype(str)
    )
    station_df = station_df.drop_duplicates(subset=["station_id"]).reset_index(drop=True)
    station_df["station_idx"] = np.arange(station_df.shape[0], dtype=np.int32)
    return station_df


def load_station_obs_matrix(time_axis: pd.DatetimeIndex, station_df: pd.DataFrame) -> np.ndarray:
    con = duckdb.connect()
    con.register("station_inventory", station_df[["station_id", "station_idx"]])
    query = f"""
        SELECT
            CAST(a.hour_start_utc AS TIMESTAMP) AS time_utc,
            s.station_idx,
            CAST(a.accum_mm AS DOUBLE) AS accum_mm
        FROM read_parquet('{AVAMET_HOURLY}') AS a
        JOIN station_inventory AS s USING (station_id)
        WHERE a.complete_strict
          AND CAST(a.hour_start_utc AS TIMESTAMP) >= ?
          AND CAST(a.hour_start_utc AS TIMESTAMP) < ?
        ORDER BY time_utc, station_idx
    """
    station_obs = con.execute(query, [TIME_START, TIME_END]).fetchdf()
    con.close()
    station_obs["time_utc"] = pd.to_datetime(station_obs["time_utc"], utc=False)

    time_indexer = pd.Index(time_axis)
    time_idx = time_indexer.get_indexer(station_obs["time_utc"])
    station_idx = station_obs["station_idx"].to_numpy(dtype=np.int32, copy=False)
    obs_matrix = np.full((time_axis.size, station_df.shape[0]), np.nan, dtype=np.float32)
    valid_mask = time_idx >= 0
    obs_matrix[time_idx[valid_mask], station_idx[valid_mask]] = station_obs.loc[valid_mask, "accum_mm"].to_numpy(
        dtype=np.float32
    )
    return obs_matrix


def build_neighbor_rank_tables(static_df: pd.DataFrame, station_df: pd.DataFrame, max_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    cell_lat = static_df["lat"].to_numpy(dtype=np.float64, copy=False)
    cell_lon = static_df["lon"].to_numpy(dtype=np.float64, copy=False)
    station_lat = station_df["lat"].to_numpy(dtype=np.float64, copy=False)
    station_lon = station_df["lon"].to_numpy(dtype=np.float64, copy=False)
    distance_km = haversine_km(cell_lat, cell_lon, station_lat, station_lon)

    cell_grid_ids = static_df["grid_id"].to_numpy()
    station_grid_ids = station_df["grid_id"].to_numpy()
    distance_km[cell_grid_ids[:, None] == station_grid_ids[None, :]] = np.inf

    neighbor_idx = np.full((static_df.shape[0], max_neighbors), -1, dtype=np.int32)
    neighbor_dist = np.full((static_df.shape[0], max_neighbors), np.nan, dtype=np.float32)
    for cell_idx in range(static_df.shape[0]):
        order = np.argsort(distance_km[cell_idx])[:max_neighbors]
        finite = np.isfinite(distance_km[cell_idx, order])
        keep = order[finite]
        neighbor_idx[cell_idx, : keep.size] = keep
        neighbor_dist[cell_idx, : keep.size] = distance_km[cell_idx, keep].astype(np.float32)
    return neighbor_idx, neighbor_dist


def build_neighbor_variant(
    base_idx: np.ndarray,
    base_dist: np.ndarray,
    k: int,
    radius_km: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    idx = base_idx[:, :k].copy()
    dist = base_dist[:, :k].copy()
    if radius_km is not None:
        too_far = np.isfinite(dist) & (dist > radius_km)
        idx[too_far] = -1
        dist[too_far] = np.nan
    return idx, dist


def predict_idw_matrix(obs_matrix: np.ndarray, neighbor_idx: np.ndarray, neighbor_dist: np.ndarray) -> np.ndarray:
    n_hours = obs_matrix.shape[0]
    n_cells = neighbor_idx.shape[0]
    out = np.full((n_hours, n_cells), np.nan, dtype=np.float32)
    for cell_idx in range(n_cells):
        idx = neighbor_idx[cell_idx]
        valid_neighbors = idx >= 0
        idx = idx[valid_neighbors]
        if idx.size == 0:
            continue
        dist = np.clip(neighbor_dist[cell_idx, valid_neighbors].astype(np.float32), 0.1, None)
        weights = (1.0 / np.power(dist, IDW_POWER)).astype(np.float32)
        values = obs_matrix[:, idx]
        active = np.isfinite(values)
        denom = (active * weights[None, :]).sum(axis=1, dtype=np.float32)
        numer = (np.where(active, values, 0.0) * weights[None, :]).sum(axis=1, dtype=np.float32)
        pred = np.full(n_hours, np.nan, dtype=np.float32)
        mask = denom > 0
        pred[mask] = numer[mask] / denom[mask]
        out[:, cell_idx] = np.clip(pred, a_min=0.0, a_max=None)
    return out


def extract_row_predictions(matrix: np.ndarray, time_idx: np.ndarray, cell_idx: np.ndarray) -> np.ndarray:
    return matrix[time_idx, cell_idx].astype(np.float32, copy=False)


def build_prediction_frame() -> pd.DataFrame:
    df = add_context_columns(load_common_overlap_2023())

    static_df = pd.read_parquet(STATIC_PARQUET).copy()
    static_df = static_df.sort_values(["imerg_local_lon_idx", "imerg_local_lat_idx"]).reset_index(drop=True)
    static_df["cell_idx"] = np.arange(static_df.shape[0], dtype=np.int32)
    grid_to_idx = dict(zip(static_df["grid_id"], static_df["cell_idx"], strict=False))
    time_axis = pd.date_range(TIME_START, TIME_END - pd.Timedelta(hours=1), freq="1h")
    time_to_idx = {timestamp: idx for idx, timestamp in enumerate(time_axis)}
    df["cell_idx"] = df["grid_id"].map(grid_to_idx).astype(np.int32)
    df["time_idx"] = df["time_utc"].map(time_to_idx).astype(np.int32)

    station_df = load_station_inventory()
    obs_matrix = load_station_obs_matrix(time_axis, station_df)
    base_idx, base_dist = build_neighbor_rank_tables(static_df, station_df, max_neighbors=64)
    local_variant_defs = {
        "domain_idw_leavecell_knn08": build_neighbor_variant(base_idx, base_dist, k=8),
        "domain_idw_leavecell_knn64": build_neighbor_variant(base_idx, base_dist, k=64),
        "domain_idw_leavecell_knn64_r15km": build_neighbor_variant(base_idx, base_dist, k=64, radius_km=15.0),
    }
    for baseline_name, (idx_variant, dist_variant) in local_variant_defs.items():
        matrix = predict_idw_matrix(obs_matrix, idx_variant, dist_variant)
        df[baseline_name] = extract_row_predictions(
            matrix,
            df["time_idx"].to_numpy(dtype=np.int32, copy=False),
            df["cell_idx"].to_numpy(dtype=np.int32, copy=False),
        )
    return df


def sort_candidates(metrics_by_local: dict[str, dict[str, float | int | None]]) -> tuple[str, dict[str, float | int | None]]:
    ranked = []
    for baseline_name, metrics in metrics_by_local.items():
        mae = metrics["mae_mm"]
        rmse = metrics["rmse_mm"]
        ranked.append((np.inf if mae is None else float(mae), np.inf if rmse is None else float(rmse), baseline_name))
    _, _, best_name = min(ranked)
    return best_name, metrics_by_local[best_name]


def selection_label(scenario: str, regime_stratifier: str) -> str:
    if scenario == "robust_ge_2" and regime_stratifier in {"overall", "observable_conflict", "target_intensity"}:
        return "primary"
    return "sensitivity"


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    y_true = df["avamet_agg_mm"].to_numpy(dtype=np.float32, copy=False)
    predictions = {}
    for baseline, col in EO_COLUMNS.items():
        predictions[baseline] = df[col].to_numpy(dtype=np.float32, copy=False)
    for baseline in LOCAL_BASELINES:
        predictions[baseline] = df[baseline].to_numpy(dtype=np.float32, copy=False)

    rows: list[dict[str, object]] = []
    support_bins = ["all"] + [b for b in df["support_score_bin"].dropna().astype(str).unique().tolist() if b != "unknown"]
    support_bins = sorted(support_bins, key=lambda x: SUPPORT_BIN_ORDER.get(x, 99))

    for scenario_name, min_support in SCENARIOS.items():
        scenario_mask = df["avamet_n_active"].to_numpy(dtype=np.float32, copy=False) >= float(min_support)
        if not np.any(scenario_mask):
            continue

        for support_bin in support_bins:
            if support_bin == "all":
                support_mask = scenario_mask
            else:
                support_mask = scenario_mask & (df["support_score_bin"].astype(str).to_numpy() == support_bin)
            if not np.any(support_mask):
                continue

            regime_masks = {
                "overall": {"all": support_mask},
                "observable_conflict": {
                    str(value): support_mask & (df["observable_conflict_bin"].astype(str).to_numpy() == str(value))
                    for value in ["low_le_p75", "mid_p75_p95", "tail_gt_p95"]
                },
                "target_intensity": {
                    str(value): support_mask & (df["target_intensity_bin_v2"].astype(str).to_numpy() == str(value))
                    for value in ["=0", "(0,0.1]", "(0.1,1]", "(1,5]", "(5,10]", "(10,20]", ">20"]
                },
            }

            for regime_stratifier in KEEP_REGIME_STRATIFIERS:
                for regime_stratum, mask in regime_masks[regime_stratifier].items():
                    if not np.any(mask):
                        continue

                    subset_y = y_true[mask]
                    subset_df = df.loc[mask]
                    fixed_local_metrics = compute_metrics(subset_y, predictions[FIXED_LOCAL_BASELINE][mask])
                    local_metrics_by_name = {
                        baseline_name: compute_metrics(subset_y, predictions[baseline_name][mask])
                        for baseline_name in LOCAL_BASELINES
                    }
                    best_local_name, best_local_metrics = sort_candidates(local_metrics_by_name)

                    for eo_name in EO_COLUMNS:
                        eo_metrics = compute_metrics(subset_y, predictions[eo_name][mask])
                        eo_mae = eo_metrics["mae_mm"]
                        fixed_mae = fixed_local_metrics["mae_mm"]
                        best_mae = best_local_metrics["mae_mm"]
                        eo_pos = eo_metrics["mae_positive_target_mm"]
                        fixed_pos = fixed_local_metrics["mae_positive_target_mm"]
                        best_pos = best_local_metrics["mae_positive_target_mm"]

                        rows.append(
                            {
                                "selection": selection_label(scenario_name, regime_stratifier),
                                "scenario": scenario_name,
                                "support_score_bin": support_bin,
                                "regime_stratifier": regime_stratifier,
                                "regime_stratum": regime_stratum,
                                "n_requested": int(mask.sum()),
                                "n_positive_target": int(np.sum(subset_y > 0.0)),
                                "mean_target_mm": float(subset_df["avamet_agg_mm"].mean()),
                                "mean_support_score": float(subset_df["avamet_support_score_v0"].mean()),
                                "mean_n_active": float(subset_df["avamet_n_active"].mean()),
                                "mean_conflict": float(subset_df["observable_conflict_nonref_v0"].mean()),
                                "eo_baseline": eo_name,
                                "eo_mae_mm": eo_mae,
                                "eo_rmse_mm": eo_metrics["rmse_mm"],
                                "eo_corr": eo_metrics["corr"],
                                "eo_mae_positive_target_mm": eo_pos,
                                "eo_coverage_share": eo_metrics["coverage_share"],
                                "fixed_local_baseline": FIXED_LOCAL_BASELINE,
                                "fixed_local_mae_mm": fixed_mae,
                                "fixed_local_rmse_mm": fixed_local_metrics["rmse_mm"],
                                "fixed_local_corr": fixed_local_metrics["corr"],
                                "fixed_local_mae_positive_target_mm": fixed_pos,
                                "fixed_local_coverage_share": fixed_local_metrics["coverage_share"],
                                "delta_mae_vs_fixed_local_mm": None if eo_mae is None or fixed_mae is None else float(eo_mae) - float(fixed_mae),
                                "delta_mae_positive_vs_fixed_local_mm": None
                                if eo_pos is None or fixed_pos is None
                                else float(eo_pos) - float(fixed_pos),
                                "best_local_baseline": best_local_name,
                                "best_local_mae_mm": best_mae,
                                "best_local_rmse_mm": best_local_metrics["rmse_mm"],
                                "best_local_corr": best_local_metrics["corr"],
                                "best_local_mae_positive_target_mm": best_pos,
                                "best_local_coverage_share": best_local_metrics["coverage_share"],
                                "delta_mae_vs_best_local_mm": None if eo_mae is None or best_mae is None else float(eo_mae) - float(best_mae),
                                "delta_mae_positive_vs_best_local_mm": None
                                if eo_pos is None or best_pos is None
                                else float(eo_pos) - float(best_pos),
                                "eo_wins_vs_fixed_local": bool(eo_mae is not None and fixed_mae is not None and eo_mae < fixed_mae),
                                "eo_wins_vs_best_local": bool(eo_mae is not None and best_mae is not None and eo_mae < best_mae),
                            }
                        )

    out = pd.DataFrame(rows)
    out["scenario_order"] = out["scenario"].map({"robust_ge_2": 0, "wide_ge_1": 1}).fillna(99)
    out["support_order"] = out["support_score_bin"].map(SUPPORT_BIN_ORDER).fillna(99)
    out["regime_stratifier_order"] = out["regime_stratifier"].map(REGIME_STRATIFIER_ORDER).fillna(99)
    out["regime_stratum_order"] = out.apply(
        lambda row: REGIME_STRATUM_ORDER.get(str(row["regime_stratifier"]), {}).get(str(row["regime_stratum"]), 99),
        axis=1,
    )
    out = out.sort_values(
        ["selection", "scenario_order", "support_order", "regime_stratifier_order", "regime_stratum_order", "eo_baseline"]
    ).reset_index(drop=True)
    return out.drop(columns=["scenario_order", "support_order", "regime_stratifier_order", "regime_stratum_order"])


def build_markdown(table_df: pd.DataFrame) -> str:
    lines = [
        "# Delta MAE by Support Score and Regime v1",
        "",
        f"- Canonical parquet: `{CANONICAL_PARQUET}`",
        f"- Fixed local baseline: `{FIXED_LOCAL_BASELINE}`",
        f"- Best-local candidate set: `{LOCAL_BASELINES}`",
        f"- EO baselines: `{list(EO_COLUMNS)}`",
        "",
        "## Primary rows worth reading first",
    ]

    primary = table_df[
        (table_df["selection"] == "primary")
        & (table_df["scenario"] == "robust_ge_2")
        & (table_df["support_score_bin"].isin(["all", "three_plus_like_gt_0.67"]))
        & (table_df["regime_stratifier"].isin(["overall", "observable_conflict"]))
    ].copy()
    if primary.empty:
        lines.append("- No primary rows were generated.")
    else:
        for _, row in primary.iterrows():
            lines.append(
                f"- `{row['scenario']}` / `{row['support_score_bin']}` / `{row['regime_stratifier']}` / "
                f"`{row['regime_stratum']}` / `{row['eo_baseline']}`: "
                f"delta vs fixed local = {row['delta_mae_vs_fixed_local_mm']:.4f}, "
                f"delta vs best local = {row['delta_mae_vs_best_local_mm']:.4f}, "
                f"delta positive-only vs best local = "
                f"{'null' if pd.isna(row['delta_mae_positive_vs_best_local_mm']) else f'{row['delta_mae_positive_vs_best_local_mm']:.4f}'}"
            )

    lines.extend(["", "## EO wins vs best local"])
    wins = (
        table_df[table_df["selection"] == "primary"]
        .groupby(["scenario", "support_score_bin", "regime_stratifier", "eo_baseline"])["eo_wins_vs_best_local"]
        .sum()
        .reset_index(name="n_wins")
    )
    wins = wins[wins["n_wins"] > 0].sort_values(
        ["scenario", "support_score_bin", "regime_stratifier", "n_wins", "eo_baseline"],
        ascending=[True, True, True, False, True],
    )
    if wins.empty:
        lines.append("- No EO baseline beats the best local baseline on MAE within the primary rows.")
    else:
        for _, row in wins.iterrows():
            lines.append(
                f"- `{row['scenario']}` / `{row['support_score_bin']}` / `{row['regime_stratifier']}`: "
                f"`{row['eo_baseline']}` wins {int(row['n_wins'])} rows"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_prediction_frame()
    table_df = build_table(df)

    csv_path = OUTPUT_DIR / "delta_mae_by_support_score_and_regime_v1.csv"
    md_path = OUTPUT_DIR / "delta_mae_by_support_score_and_regime_v1.md"
    json_path = OUTPUT_DIR / "delta_mae_by_support_score_and_regime_v1.json"

    table_df.to_csv(csv_path, index=False)
    md_path.write_text(build_markdown(table_df), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "canonical_parquet": str(CANONICAL_PARQUET),
                "static_parquet": str(STATIC_PARQUET),
                "station_csv": str(STATION_CSV),
                "avamet_hourly": str(AVAMET_HOURLY),
                "fixed_local_baseline": FIXED_LOCAL_BASELINE,
                "local_baselines": LOCAL_BASELINES,
                "eo_baselines": list(EO_COLUMNS),
                "n_rows": int(table_df.shape[0]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

