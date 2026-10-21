#!/usr/bin/env python
"""Resampling sensitivity for the CAGEO revision (R2.1).

Rebuilds 2023 ERA5 and EURADCLIM on the IMERG grid with alternative operators and re-evaluates the benchmark:
  ERA5:      bilinear (manuscript), nearest neighbour, first-order conservative (area-overlap weighted)
  EURADCLIM: footprint mean (manuscript), nearest pixel to the IMERG cell centre
Expects the regional monthly npz extracts under $BENCHMARK_REGIONAL_NPZ_DIR/{era5/cv_hourly_tp_npz,euradclim/cv_hourly_v3_npz}/<year>/.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from revision_cageo_analyses_v1 import LOCAL_COL, bootstrap_delta, conflict_slices, hour_blocks
from revision_cageo_cache_v1 import load_year


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "revision_cageo_v1"
REGIONAL_ROOT = Path(os.environ.get("BENCHMARK_REGIONAL_NPZ_DIR", ROOT / "data" / "restricted" / "regional"))
STATIC_PARQUET = ROOT / "data" / "restricted" / "imerg_cell_static_v1.parquet"
YEAR = 2023
IMERG_HALF = 0.05
ERA5_HALF = 0.125


def overlap_1d(a_lo: np.ndarray, a_hi: np.ndarray, b_lo: np.ndarray, b_hi: np.ndarray) -> np.ndarray:
    return np.clip(np.minimum(a_hi[:, None], b_hi[None, :]) - np.maximum(a_lo[:, None], b_lo[None, :]), 0.0, None)


def era5_operators(era_lon: np.ndarray, era_lat: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> dict[str, np.ndarray]:
    """Return (n_cells, n_era_lat, n_era_lon) weight tensors for each ERA5 operator."""
    n = lon.size
    # bilinear
    def axis(ax, t):
        asc = ax[0] < ax[-1]
        w = ax if asc else ax[::-1]
        hi = np.clip(np.searchsorted(w, t, side="right"), 1, w.size - 1)
        lo = hi - 1
        lo_i, hi_i = (lo, hi) if asc else (w.size - 1 - hi, w.size - 1 - lo)
        wh = np.clip(np.where(ax[hi_i] != ax[lo_i], (t - ax[lo_i]) / (ax[hi_i] - ax[lo_i]), 0.0), 0, 1)
        return lo_i, hi_i, wh
    x_lo, x_hi, wx = axis(era_lon.astype(float), lon)
    y_lo, y_hi, wy = axis(era_lat.astype(float), lat)
    bil = np.zeros((n, era_lat.size, era_lon.size))
    for i in range(n):
        bil[i, y_lo[i], x_lo[i]] += (1 - wx[i]) * (1 - wy[i])
        bil[i, y_lo[i], x_hi[i]] += wx[i] * (1 - wy[i])
        bil[i, y_hi[i], x_lo[i]] += (1 - wx[i]) * wy[i]
        bil[i, y_hi[i], x_hi[i]] += wx[i] * wy[i]
    # nearest
    nn = np.zeros_like(bil)
    iy = np.abs(era_lat[None, :] - lat[:, None]).argmin(axis=1)
    ix = np.abs(era_lon[None, :] - lon[:, None]).argmin(axis=1)
    nn[np.arange(n), iy, ix] = 1.0
    # first-order conservative: overlap area of IMERG cell with each ERA5 cell (equal-angle, small-cell approximation)
    ox = overlap_1d(lon - IMERG_HALF, lon + IMERG_HALF, era_lon - ERA5_HALF, era_lon + ERA5_HALF)
    oy = overlap_1d(lat - IMERG_HALF, lat + IMERG_HALF, era_lat - ERA5_HALF, era_lat + ERA5_HALF)
    cons = oy[:, :, None] * ox[:, None, :]
    return {"bilinear": bil, "nearest": nn, "conservative": cons}


def apply_weights(field: np.ndarray, w: np.ndarray) -> np.ndarray:
    """field (T, Y, X), w (N, Y, X) -> (T, N), renormalising over finite source values."""
    finite = np.isfinite(field)
    vals = np.where(finite, field, 0.0)
    num = np.einsum("tyx,nyx->tn", vals, w)
    den = np.einsum("tyx,nyx->tn", finite.astype(float), w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def euradclim_operators(eu_lon: np.ndarray, eu_lat: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> dict[str, list[np.ndarray]]:
    flon, flat = eu_lon.ravel(), eu_lat.ravel()
    mean_groups, nearest = [], []
    for x, y in zip(lon, lat):
        inside = (flon >= x - IMERG_HALF) & (flon < x + IMERG_HALF) & (flat >= y - IMERG_HALF) & (flat < y + IMERG_HALF)
        mean_groups.append(np.flatnonzero(inside))
        nearest.append(np.array([np.argmin((flon - x) ** 2 + ((flat - y)) ** 2)]))
    return {"footprint mean": mean_groups, "nearest": nearest}


def apply_groups(field: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    flat = field.reshape(field.shape[0], -1)
    out = np.full((field.shape[0], len(groups)), np.nan)
    for i, g in enumerate(groups):
        v = flat[:, g]
        cnt = np.isfinite(v).sum(axis=1)
        s = np.nansum(v, axis=1)
        out[:, i] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return out


def build_variants() -> pd.DataFrame:
    static = pd.read_parquet(STATIC_PARQUET).sort_values(["imerg_local_lon_idx", "imerg_local_lat_idx"]).reset_index(drop=True)
    lon = static["lon"].to_numpy(float)
    lat = static["lat"].to_numpy(float)
    frames = []
    era_ops = eu_ops = None
    for month in range(1, 13):
        er = np.load(REGIONAL_ROOT / "era5" / "cv_hourly_tp_npz" / f"{YEAR}" / f"era5_cv_tp_{YEAR}{month:02d}.npz")
        eu = np.load(REGIONAL_ROOT / "euradclim" / "cv_hourly_v3_npz" / f"{YEAR}" / f"euradclim_cv_{YEAR}{month:02d}.npz")
        if era_ops is None:
            era_ops = era5_operators(er["lon"].astype(float), er["lat"].astype(float), lon, lat)
            eu_ops = euradclim_operators(eu["lon"].astype(float), eu["lat"].astype(float), lon, lat)
        t_er = pd.to_datetime(er["times_utc"])
        t_eu = pd.to_datetime(eu["times_utc"]).round("1h")
        if not t_er.equals(t_eu):
            raise ValueError(f"time mismatch {month}")
        tp = er["tp_mm_hourly"].astype(float)
        pr = eu["precip_mm"].astype(float)
        cols = {f"era5_{k}": apply_weights(tp, w) for k, w in era_ops.items()}
        cols |= {f"eur_{k.replace(' ', '_')}": apply_groups(pr, g) for k, g in eu_ops.items()}
        n_t, n_c = tp.shape[0], lon.size
        frames.append(pd.DataFrame({
            "time_utc": np.repeat(t_er.to_numpy(), n_c),
            "grid_id": np.tile(static["grid_id"].to_numpy(), n_t),
            **{k: v.reshape(-1).astype(np.float32) for k, v in cols.items()},
        }))
        print(f"month {month:02d} done", flush=True)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    cache = OUT_DIR / "cache" / f"resampling_variants_{YEAR}.parquet"
    if not cache.exists():
        build_variants().to_parquet(cache, index=False)
    var = pd.read_parquet(cache)
    full = load_year(YEAR)
    full = full.merge(var, on=["time_utc", "grid_id"], how="left")
    check = {
        "era5_bilinear_max_abs_diff": float(np.nanmax(np.abs(full["era5_bilinear"] - full["era5_mm"]))),
        "eur_footprint_mean_max_abs_diff": float(np.nanmax(np.abs(full["eur_footprint_mean"] - full["euradclim_on_imerg_mm"]))),
    }
    print(check)

    configs = {
        "manuscript (ERA5 bilinear, EUR footprint mean)": ("era5_bilinear", "eur_footprint_mean"),
        "ERA5 nearest": ("era5_nearest", "eur_footprint_mean"),
        "ERA5 conservative": ("era5_conservative", "eur_footprint_mean"),
        "EURADCLIM nearest": ("era5_bilinear", "eur_nearest"),
        "both nearest": ("era5_nearest", "eur_nearest"),
    }
    rows = []
    for cfg, (ecol, ucol) in configs.items():
        f = full.copy()
        f["era5_v"], f["eur_v"] = f[ecol], f[ucol]
        f = f.loc[np.isfinite(f["era5_v"]) & np.isfinite(f["eur_v"])]
        p = f[["imerg_mm", "era5_v", "eur_v"]].to_numpy()
        f["kappa_v"] = (np.abs(p[:, 0] - p[:, 1]) + np.abs(p[:, 0] - p[:, 2]) + np.abs(p[:, 1] - p[:, 2])) / 3.0
        ref = f["kappa_v"]
        rob = f.loc[f["avamet_n_active"] >= 2].reset_index(drop=True)
        y = rob["avamet_agg_mm"].to_numpy(float)
        loc = rob[LOCAL_COL].to_numpy(float)
        codes = hour_blocks(rob)
        cb, _ = conflict_slices(rob, col="kappa_v", ref=ref)
        slices = {"all hours": np.ones(y.size, bool), "positive rain": y > 0, **{f"{k} conflict": v for k, v in cb.items()}}
        for sname, m in slices.items():
            for name, col in [("IMERG", "imerg_mm"), ("ERA5", "era5_v"), ("EURADCLIM", "eur_v")]:
                pr = rob[col].to_numpy(float)
                res = bootstrap_delta(y[m], pr[m], loc[m], codes[m])
                rows.append({"config": cfg, "slice": sname, "source": name, "n": int(m.sum()),
                             "mae": float(np.mean(np.abs(pr[m] - y[m]))), **res})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "resampling_sensitivity_2023.csv", index=False)

    # magnitude of resampling-induced differences on robust rows
    rob = full.loc[full["avamet_n_active"] >= 2]
    diffs = []
    for a, b in [("era5_bilinear", "era5_nearest"), ("era5_bilinear", "era5_conservative"), ("eur_footprint_mean", "eur_nearest")]:
        d = (rob[a] - rob[b]).abs()
        wet = rob["avamet_agg_mm"] > 0
        diffs.append({"pair": f"{a} vs {b}", "mean_abs_diff_all": float(d.mean()), "mean_abs_diff_wet": float(d[wet].mean()),
                      "corr_wet": float(np.corrcoef(rob.loc[wet, a], rob.loc[wet, b])[0, 1])})
    pd.DataFrame(diffs).to_csv(OUT_DIR / "resampling_operator_differences_2023.csv", index=False)


if __name__ == "__main__":
    main()
