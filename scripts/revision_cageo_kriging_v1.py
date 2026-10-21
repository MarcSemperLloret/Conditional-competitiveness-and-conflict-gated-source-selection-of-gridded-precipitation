#!/usr/bin/env python
"""Geostatistical local comparators for the CAGEO revision (R2.2).

Leave-cell ordinary kriging (OK) and kriging with external drift on elevation (KED), built on exactly the same
64 nearest out-of-cell gauges as the fixed IDW comparator. A climatological standardized exponential variogram
(nugget + exponential, unit sill) is fitted on 2022 wet hours (raw values for OK, elevation-detrended residuals
for KED) and applied unchanged to 2023. Hours in which no gauge reports rain are predicted as zero.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import build_delta_mae_by_support_score_and_regime_v1 as base
from revision_cageo_analyses_v1 import (
    EO_SOURCES, INTENSITY_BINS, INTENSITY_LABELS, LOCAL_COL, SOURCES, bootstrap_delta, conflict_slices, hour_blocks,
    spell_blocks,
)
from revision_cageo_cache_v1 import load_year


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "revision_cageo_v1"
FIT_YEAR = 2022
EVAL_YEAR = 2023
MAX_LAG_KM = 60.0
LAG_BIN_KM = 2.0
MIN_WET_STATIONS = 30
K = 64


def set_year(year: int) -> pd.DatetimeIndex:
    base.TIME_START = pd.Timestamp(f"{year}-01-01")
    base.TIME_END = pd.Timestamp(f"{year + 1}-01-01")
    return pd.date_range(base.TIME_START, base.TIME_END - pd.Timedelta(hours=1), freq="1h")


def exp_model(h: np.ndarray, nugget: float, range_km: float) -> np.ndarray:
    return nugget + (1.0 - nugget) * (1.0 - np.exp(-h / range_km))


def fit_variogram(obs: np.ndarray, stations: pd.DataFrame, detrend_elev: bool) -> dict[str, float]:
    d = base.haversine_km(stations["lat"].to_numpy(), stations["lon"].to_numpy(), stations["lat"].to_numpy(), stations["lon"].to_numpy())
    iu, ju = np.triu_indices(d.shape[0], k=1)
    keep = d[iu, ju] <= MAX_LAG_KM
    iu, ju, dist = iu[keep], ju[keep], d[iu, ju][keep]
    lag = np.minimum((dist // LAG_BIN_KM).astype(int), int(MAX_LAG_KM / LAG_BIN_KM) - 1)
    nb = int(MAX_LAG_KM / LAG_BIN_KM)
    sums, counts = np.zeros(nb), np.zeros(nb)
    elev = stations["alt"].to_numpy(float) / 1000.0
    n_hours = 0
    for row in obs:
        act = np.isfinite(row)
        if (row[act] > 0).sum() < MIN_WET_STATIONS:
            continue
        v = row.copy()
        if detrend_elev:
            A = np.column_stack([np.ones(act.sum()), elev[act]])
            coef, *_ = np.linalg.lstsq(A, v[act], rcond=None)
            v[act] = v[act] - A @ coef
        sd = np.nanstd(v[act])
        if sd <= 0:
            continue
        z = (v - np.nanmean(v[act])) / sd
        ok = act[iu] & act[ju]
        sq = 0.5 * (z[iu[ok]] - z[ju[ok]]) ** 2
        sums += np.bincount(lag[ok], sq, nb)
        counts += np.bincount(lag[ok], minlength=nb)
        n_hours += 1
    centres = (np.arange(nb) + 0.5) * LAG_BIN_KM
    gamma = sums / np.maximum(counts, 1)
    (nug, rng_km), _ = curve_fit(exp_model, centres, gamma, p0=[0.2, 15.0], bounds=([0.0, 0.5], [0.99, 500.0]),
                                 sigma=1.0 / np.sqrt(np.maximum(counts, 1)))
    pd.DataFrame({"lag_km": centres, "gamma": gamma, "pairs": counts, "model": exp_model(centres, nug, rng_km)}).to_csv(
        OUT_DIR / f"variogram_{'ked' if detrend_elev else 'ok'}_{FIT_YEAR}.csv", index=False)
    return {"nugget": float(nug), "range_km": float(rng_km), "n_hours": n_hours}


def krige_matrix(obs: np.ndarray, stations: pd.DataFrame, static: pd.DataFrame, vg: dict[str, float], drift: bool) -> np.ndarray:
    idx, dist = base.build_neighbor_rank_tables(static, stations, max_neighbors=K)
    lat, lon = stations["lat"].to_numpy(), stations["lon"].to_numpy()
    n_cells = idx.shape[0]
    safe = np.where(idx >= 0, idx, 0)
    # neighbour-neighbour distances per cell
    dnn = np.stack([base.haversine_km(lat[safe[c]], lon[safe[c]], lat[safe[c]], lon[safe[c]]) for c in range(n_cells)])
    cov = lambda h: np.where(h > 1e-9, (1.0 - vg["nugget"]) * np.exp(-h / vg["range_km"]), 1.0)
    C = cov(dnn)  # (cells, K, K)
    c0 = cov(np.nan_to_num(dist, nan=1e6))  # (cells, K)
    elev_st = stations["alt"].to_numpy(float) / 1000.0
    f_nb = elev_st[safe]  # (cells, K)
    f0 = static["subgrid_elevation_mean_m"].to_numpy(float) / 1000.0
    m = K + (2 if drift else 1)

    out = np.zeros((obs.shape[0], n_cells), dtype=np.float32)
    for t in range(obs.shape[0]):
        vals = obs[t][safe]
        act = (idx >= 0) & np.isfinite(vals)
        if not np.any(np.nan_to_num(obs[t]) > 0):
            out[t] = np.where(act.any(axis=1), 0.0, np.nan)
            continue
        A = np.zeros((n_cells, m, m))
        b = np.zeros((n_cells, m))
        A[:, :K, :K] = C * (act[:, :, None] & act[:, None, :])
        diag = np.arange(K)
        A[:, diag, diag] = np.where(act, 1.0 + 1e-6, 1.0)
        A[:, :K, K] = act
        A[:, K, :K] = act
        b[:, :K] = c0 * act
        b[:, K] = 1.0
        use_drift = drift
        if drift:
            n_act = act.sum(axis=1)
            spread = np.array([np.ptp(f_nb[c][act[c]]) if n_act[c] else 0.0 for c in range(n_cells)])
            ok_drift = (n_act >= 4) & (spread > 0.01)
            A[:, :K, K + 1] = np.where(ok_drift[:, None], f_nb * act, 0.0)
            A[:, K + 1, :K] = A[:, :K, K + 1]
            A[:, K + 1, K + 1] = np.where(ok_drift, 0.0, 1.0)  # decouple drift row when not usable -> OK
            b[:, K + 1] = np.where(ok_drift, f0, 0.0)
        try:
            w = np.linalg.solve(A, b[..., None])[..., 0]
        except np.linalg.LinAlgError:
            w = np.einsum("nij,nj->ni", np.linalg.pinv(A), b)
        pred = (w[:, :K] * np.where(act, vals, 0.0)).sum(axis=1)
        pred = np.where(act.any(axis=1), np.clip(pred, 0.0, None), np.nan)
        out[t] = pred
        if t % 1000 == 0:
            print(f"  hour {t}", flush=True)
    return out


def evaluate(df: pd.DataFrame, local_cols: dict[str, str]) -> pd.DataFrame:
    full = df
    rob = df.loc[df["avamet_n_active"] >= 2].reset_index(drop=True)
    y = rob["avamet_agg_mm"].to_numpy(float)
    codes = {"hour": hour_blocks(rob), "wet-spell event": spell_blocks(rob)}
    cb, _ = conflict_slices(rob, ref=full["observable_conflict_nonref_v0"])
    ib = pd.cut(rob["avamet_agg_mm"], INTENSITY_BINS, labels=INTENSITY_LABELS).astype(str).to_numpy()
    slices = {"all hours": np.ones(y.size, bool), "positive rain": y > 0, **{f"{k} conflict": v for k, v in cb.items()},
              **{f"rain {b}": ib == b for b in INTENSITY_LABELS}}
    rows = []
    for lname, lcol in local_cols.items():
        loc = rob[lcol].to_numpy(float)
        for sname, msk in slices.items():
            m = msk & np.isfinite(loc)
            base_row = {"local": lname, "slice": sname, "n": int(m.sum()), "mae_local": float(np.mean(np.abs(loc[m] - y[m])))}
            for eo in EO_SOURCES:
                p = rob[SOURCES[eo]].to_numpy(float)
                for bname, cd in codes.items():
                    if bname != "hour" and sname not in ("all hours", "positive rain"):
                        continue
                    rows.append({**base_row, "source": eo, "block": bname, **bootstrap_delta(y[m], p[m], loc[m], cd[m])})
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / "cache" / f"kriging_predictions_{EVAL_YEAR}.parquet"
    static = pd.read_parquet(base.STATIC_PARQUET).sort_values(["imerg_local_lon_idx", "imerg_local_lat_idx"]).reset_index(drop=True)
    if not cache.exists():
        stations = base.load_station_inventory()
        fit_axis = set_year(FIT_YEAR)
        obs_fit = base.load_station_obs_matrix(fit_axis, stations)
        vg_ok = fit_variogram(obs_fit, stations, detrend_elev=False)
        vg_ked = fit_variogram(obs_fit, stations, detrend_elev=True)
        print("variogram OK", vg_ok, "KED", vg_ked, flush=True)
        pd.DataFrame([{"model": "OK", **vg_ok}, {"model": "KED", **vg_ked}]).to_csv(OUT_DIR / "variogram_params_2022.csv", index=False)

        eval_axis = set_year(EVAL_YEAR)
        obs = base.load_station_obs_matrix(eval_axis, stations)
        print("OK ...", flush=True)
        ok = krige_matrix(obs, stations, static, vg_ok, drift=False)
        print("KED ...", flush=True)
        ked = krige_matrix(obs, stations, static, vg_ked, drift=True)
        n_t, n_c = ok.shape
        pd.DataFrame({
            "time_utc": np.repeat(eval_axis.to_numpy(), n_c),
            "grid_id": np.tile(static["grid_id"].to_numpy(), n_t),
            "local_ok_leavecell_knn64": ok.reshape(-1),
            "local_ked_elev_leavecell_knn64": ked.reshape(-1),
        }).to_parquet(cache, index=False)

    df = load_year(EVAL_YEAR).merge(pd.read_parquet(cache), on=["time_utc", "grid_id"], how="left")
    res = evaluate(df, {
        "IDW-64 (manuscript)": LOCAL_COL,
        "OK-64": "local_ok_leavecell_knn64",
        "KED-64 (elevation)": "local_ked_elev_leavecell_knn64",
    })
    res.to_csv(OUT_DIR / "kriging_comparator_sensitivity_2023.csv", index=False)


if __name__ == "__main__":
    main()
