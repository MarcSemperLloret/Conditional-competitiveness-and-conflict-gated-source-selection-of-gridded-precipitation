#!/usr/bin/env python
"""Within-cell target representativeness diagnostics for the CAGEO revision (R2.4).

For every robust-support 2023 cell-hour, reconstructs the active within-cell gauge set and derives
  - mean distance of active gauges to the IMERG cell centre,
  - maximum inter-gauge distance and elevation range of the active set,
  - median-minus-mean target difference,
  - a split-half target noise floor (cell-hours with >= 4 active gauges): MAE between the medians of two
    disjoint random halves of the active gauges, compared with gridded and local errors on the same rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import build_delta_mae_by_support_score_and_regime_v1 as base
from revision_cageo_analyses_v1 import EO_SOURCES, LOCAL_COL, SOURCES, bootstrap_delta, hour_blocks
from revision_cageo_cache_v1 import load_year


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "revision_cageo_v1"
YEAR = 2023
SEED = 42


def station_geometry() -> tuple[pd.DataFrame, np.ndarray]:
    base.TIME_START = pd.Timestamp(f"{YEAR}-01-01")
    base.TIME_END = pd.Timestamp(f"{YEAR + 1}-01-01")
    stations = base.load_station_inventory()
    time_axis = pd.date_range(base.TIME_START, base.TIME_END - pd.Timedelta(hours=1), freq="1h")
    obs = base.load_station_obs_matrix(time_axis, stations)
    return stations, obs


def per_row_geometry(df: pd.DataFrame, stations: pd.DataFrame, obs: np.ndarray) -> pd.DataFrame:
    static = pd.read_parquet(base.STATIC_PARQUET).set_index("grid_id")
    rng = np.random.default_rng(SEED)
    t_axis = pd.date_range(base.TIME_START, base.TIME_END - pd.Timedelta(hours=1), freq="1h")
    t_idx = pd.Index(t_axis).get_indexer(pd.to_datetime(df["time_utc"]))
    out = {k: np.full(len(df), np.nan) for k in [
        "n_check", "median_check", "mean_target", "mean_dist_centre_km", "max_pair_km", "alt_range_m",
        "split_half_abs_diff"]}
    for grid_id, rows in df.groupby("grid_id").indices.items():
        st = stations.loc[stations["grid_id"] == grid_id]
        if st.empty:
            continue
        clat, clon = static.loc[grid_id, ["lat", "lon"]]
        d_centre = base.haversine_km(np.array([clat]), np.array([clon]), st["lat"].to_numpy(), st["lon"].to_numpy())[0]
        d_pair = base.haversine_km(st["lat"].to_numpy(), st["lon"].to_numpy(), st["lat"].to_numpy(), st["lon"].to_numpy())
        alt = st["alt"].to_numpy(dtype=float)
        vals = obs[np.ix_(t_idx[rows], st["station_idx"].to_numpy())]
        active = np.isfinite(vals)
        # group rows by identical active pattern so geometry/splits are computed once per pattern
        patterns, inv = np.unique(active, axis=0, return_inverse=True)
        for p_i, pat in enumerate(patterns):
            sel = np.where(inv.ravel() == p_i)[0]
            r = rows[sel]
            n = int(pat.sum())
            if n == 0:
                continue
            v = vals[sel][:, pat]
            out["n_check"][r] = n
            out["median_check"][r] = np.median(v, axis=1)
            out["mean_target"][r] = v.mean(axis=1)
            out["mean_dist_centre_km"][r] = d_centre[pat].mean()
            out["max_pair_km"][r] = d_pair[np.ix_(pat, pat)].max()
            out["alt_range_m"][r] = np.nanmax(alt[pat]) - np.nanmin(alt[pat])
            if n >= 4:
                perm = rng.permutation(n)
                a, b = perm[: n // 2], perm[n // 2 : 2 * (n // 2)]
                out["split_half_abs_diff"][r] = np.abs(np.median(v[:, a], axis=1) - np.median(v[:, b], axis=1))
    for k, v in out.items():
        df[k] = v
    return df


def tercile_masks(x: np.ndarray, valid: np.ndarray, label: str) -> dict[str, np.ndarray]:
    q1, q2 = np.nanquantile(x[valid], [1 / 3, 2 / 3])
    return {
        f"{label} T1 (<= {q1:.1f})": valid & (x <= q1),
        f"{label} T2": valid & (x > q1) & (x <= q2),
        f"{label} T3 (> {q2:.1f})": valid & (x > q2),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_year(YEAR)
    df = df.loc[df["avamet_n_active"] >= 2].reset_index(drop=True)
    stations, obs = station_geometry()
    df = per_row_geometry(df, stations, obs)

    y = df["avamet_agg_mm"].to_numpy(np.float64)
    loc = df[LOCAL_COL].to_numpy(np.float64)
    codes = hour_blocks(df)
    check = {
        "rows": len(df),
        "n_active_match_share": float(np.mean(df["n_check"] == df["avamet_n_active"])),
        "median_match_share": float(np.mean(np.isclose(df["median_check"], y, atol=1e-4))),
    }
    print(check)

    pos = y > 0
    n_act = df["avamet_n_active"].to_numpy()
    strat = {
        "n gauges = 2": n_act == 2,
        "n gauges = 3": n_act == 3,
        "n gauges >= 4": n_act >= 4,
    }
    strat |= tercile_masks(df["mean_dist_centre_km"].to_numpy(), np.ones(len(df), bool), "dist to centre km")
    strat |= tercile_masks(df["max_pair_km"].to_numpy(), np.ones(len(df), bool), "max inter-gauge km")
    strat |= tercile_masks(df["alt_range_m"].to_numpy(), np.ones(len(df), bool), "gauge elevation range m")
    rows = []
    for name, m in strat.items():
        m = m & pos
        for eo in EO_SOURCES:
            p = df[SOURCES[eo]].to_numpy(np.float64)
            res = bootstrap_delta(y[m], p[m], loc[m], codes[m])
            rows.append({"stratum": name, "source": eo, "n_pos": int(m.sum()),
                         "mae_local": float(np.mean(np.abs(loc[m] - y[m]))), **res})
    pd.DataFrame(rows).to_csv(OUT_DIR / "representativeness_geometry_2023.csv", index=False)

    # median vs mean and split-half noise floor
    mean_t = df["mean_target"].to_numpy()
    sh = df["split_half_abs_diff"].to_numpy()
    has_sh = np.isfinite(sh)
    floor = []
    for name, m in {"all hours": np.ones(len(df), bool), "positive rain": pos}.items():
        rec = {
            "slice": name,
            "n": int(m.sum()),
            "mad_median_vs_mean_mm": float(np.mean(np.abs(y[m] - mean_t[m]))),
            "rel_median_vs_mean": float(np.sum(np.abs(y[m] - mean_t[m])) / np.sum(mean_t[m])) if mean_t[m].sum() > 0 else np.nan,
            "n_split_half": int((m & has_sh).sum()),
            "split_half_mae_mm": float(np.mean(sh[m & has_sh])),
            "local_mae_same_rows_mm": float(np.mean(np.abs(loc - y)[m & has_sh])),
        }
        for eo in EO_SOURCES:
            p = df[SOURCES[eo]].to_numpy(np.float64)
            rec[f"{eo}_mae_same_rows_mm"] = float(np.mean(np.abs(p - y)[m & has_sh]))
            # Delta MAE with mean target instead of median
            rec[f"{eo}_delta_mae_mean_target"] = float(np.mean(np.abs(p - mean_t)[m]) - np.mean(np.abs(loc - mean_t)[m]))
            rec[f"{eo}_delta_mae_median_target"] = float(np.mean(np.abs(p - y)[m]) - np.mean(np.abs(loc - y)[m]))
        floor.append(rec)
    pd.DataFrame(floor).to_csv(OUT_DIR / "representativeness_noise_floor_2023.csv", index=False)
    df[["time_utc", "grid_id", "avamet_n_active", "mean_dist_centre_km", "max_pair_km", "alt_range_m", "mean_target",
        "split_half_abs_diff"]].to_parquet(OUT_DIR / "cache" / "representativeness_rows_2023.parquet", index=False)


if __name__ == "__main__":
    main()
