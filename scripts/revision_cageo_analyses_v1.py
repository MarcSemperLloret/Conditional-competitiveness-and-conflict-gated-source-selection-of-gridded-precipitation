#!/usr/bin/env python
"""CAGEO-D-26-00838 revision analyses on the cached robust-support prediction frames.

Blocks:
  metrics      multi-metric table (MAE, RMSE, bias, r, KGE, POD/FAR/CSI) for products and fixed local (R2.5)
  bootstrap    block-length and wet-spell (event) bootstrap sensitivity of Delta MAE (R2.6)
  conflict     normalized conflict index, conflict x intensity, Spearman diagnostics (R2.3, R1.3)
  thresholds   alternative conflict percentile cut-offs (R1.2)
  years        per-year Delta MAE, wet share, extremes and routing transfer 2019-2023 (R2.8, Editor)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import build_conflict_policy_extension_v1 as ext
from revision_cageo_cache_v1 import YEARS, load_year


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "revision_cageo_v1"
MAIN_YEAR = 2023
N_BOOTSTRAP = 1500
SEED = 42
LOCAL_COL = "domain_idw_leavecell_knn64"
SOURCES = {
    "IMERG": "imerg_mm",
    "ERA5": "era5_mm",
    "EURADCLIM": "euradclim_on_imerg_mm",
    "Local IDW-64": LOCAL_COL,
}
EO_SOURCES = ["IMERG", "ERA5", "EURADCLIM"]
ALERT_THRESHOLDS_MM = [0.1, 1.0, 5.0, 10.0]
REL_CONFLICT_EPS_MM = 0.01  # kappa / (mean + eps) is then a scale-free dispersion in [0, 2]
SPELL_GAP_HOURS = 6
INTENSITY_BINS = [0.0, 1.0, 5.0, 10.0, np.inf]
INTENSITY_LABELS = ["(0,1]", "(1,5]", "(5,10]", ">10"]


def robust_frame(year: int) -> pd.DataFrame:
    df = load_year(year)
    df = df.loc[df["avamet_n_active"] >= 2].reset_index(drop=True)
    return df


# ---------------------------------------------------------------- metrics

def kge(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float, float]:
    r = float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else np.nan
    alpha = float(np.std(p) / np.std(y)) if np.std(y) > 0 else np.nan
    beta = float(np.mean(p) / np.mean(y)) if np.mean(y) > 0 else np.nan
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)), r, alpha, beta


def continuous_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    err = p - y
    k, r, alpha, beta = kge(y, p)
    return {
        "mae_mm": float(np.mean(np.abs(err))),
        "rmse_mm": float(np.sqrt(np.mean(err**2))),
        "bias_mm": float(np.mean(err)),
        "pbias_pct": float(100.0 * np.sum(err) / np.sum(y)) if np.sum(y) > 0 else np.nan,
        "r": r,
        "kge": k,
        "kge_alpha": alpha,
        "kge_beta": beta,
    }


def run_metrics() -> pd.DataFrame:
    df = robust_frame(MAIN_YEAR)
    y = df["avamet_agg_mm"].to_numpy(np.float64)
    rows = []
    slices = {"all hours": np.ones(y.size, bool), "positive rain": y > 0}
    for slice_name, mask in slices.items():
        for name, col in SOURCES.items():
            p = df[col].to_numpy(np.float64)
            row = {"slice": slice_name, "source": name, "n": int(mask.sum()), **continuous_metrics(y[mask], p[mask])}
            if slice_name == "all hours":
                for thr in ALERT_THRESHOLDS_MM:
                    trip = ext.csi_triplet(y, p, thr)
                    row[f"pod_{thr:g}mm"] = trip["pod"]
                    row[f"far_{thr:g}mm"] = trip["far"]
                    row[f"csi_{thr:g}mm"] = trip["csi"]
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "metrics_multi_2023.csv", index=False)

    # Block-bootstrap CI for Delta RMSE (gridded minus local), hour and event blocks
    blocks = {"hour": hour_blocks(df), "wet-spell event": spell_blocks(df)}
    loc = df[LOCAL_COL].to_numpy(np.float64)
    rmse_rows = []
    for slice_name, mask in slices.items():
        for eo in EO_SOURCES:
            p = df[SOURCES[eo]].to_numpy(np.float64)
            for block_name, codes in blocks.items():
                res = bootstrap_delta(y[mask], p[mask], loc[mask], codes[mask], squared=True)
                rmse_rows.append({"slice": slice_name, "source": eo, "block": block_name, **res})
    pd.DataFrame(rmse_rows).to_csv(OUT_DIR / "delta_rmse_bootstrap_2023.csv", index=False)
    return out


# ---------------------------------------------------------------- bootstrap

def hour_blocks(df: pd.DataFrame) -> np.ndarray:
    codes, _ = pd.factorize(df["time_utc"], sort=True)
    return codes.astype(np.int64)


def calendar_blocks(df: pd.DataFrame, days: int) -> np.ndarray:
    t = pd.to_datetime(df["time_utc"])
    origin = pd.Timestamp(f"{t.min().year}-01-01")
    return ((t - origin).dt.days // days).to_numpy(np.int64)


def spell_table(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-level wet spells: hours with any robust cell target > 0, merged across gaps <= SPELL_GAP_HOURS.

    Each wet spell is one block; each dry interval between spells is one block.
    """
    t = pd.to_datetime(df["time_utc"])
    wet_hours = pd.Index(np.sort(t[df["avamet_agg_mm"].to_numpy() > 0].unique()))
    all_hours = pd.Index(np.sort(t.unique()))
    hourly = pd.Series(all_hours.isin(wet_hours), index=all_hours)
    # spell ids over wet hours with gap merging
    wet_t = hourly.index[hourly.to_numpy()]
    gaps = np.diff(wet_t.values).astype("timedelta64[h]").astype(np.int64)
    new_spell = np.concatenate([[True], gaps > SPELL_GAP_HOURS])
    spell_id = np.cumsum(new_spell) - 1
    spells = pd.DataFrame({"time": wet_t, "spell": spell_id})
    bounds = spells.groupby("spell")["time"].agg(["min", "max"])
    bounds["length_h"] = (bounds["max"] - bounds["min"]).dt.total_seconds() / 3600 + 1
    return bounds


def spell_blocks(df: pd.DataFrame) -> np.ndarray:
    bounds = spell_table(df)
    t = pd.to_datetime(df["time_utc"]).to_numpy()
    starts = bounds["min"].to_numpy()
    ends = bounds["max"].to_numpy()
    k = np.searchsorted(starts, t, side="right") - 1  # last spell starting at or before t
    in_spell = (k >= 0) & (t <= ends[np.clip(k, 0, None)])
    # wet spell k -> block 2k+1; dry interval after spell k (or before first, k=-1) -> block 2k+2
    codes = np.where(in_spell, 2 * k + 1, 2 * k + 2)
    _, codes = np.unique(codes, return_inverse=True)
    return codes.astype(np.int64)


def bootstrap_delta(
    y: np.ndarray, p_eo: np.ndarray, p_loc: np.ndarray, codes: np.ndarray, squared: bool = False
) -> dict[str, float]:
    e_eo = (p_eo - y) ** 2 if squared else np.abs(p_eo - y)
    e_lo = (p_loc - y) ** 2 if squared else np.abs(p_loc - y)
    _, codes = np.unique(codes, return_inverse=True)
    nb = int(codes.max()) + 1
    s_eo = np.bincount(codes, e_eo, nb)
    s_lo = np.bincount(codes, e_lo, nb)
    cnt = np.bincount(codes, minlength=nb).astype(np.float64)
    f = np.sqrt if squared else (lambda x: x)
    point = float(f(s_eo.sum() / cnt.sum()) - f(s_lo.sum() / cnt.sum()))
    rng = np.random.default_rng(SEED)
    boot = np.empty(N_BOOTSTRAP)
    chunk = 100
    for start in range(0, N_BOOTSTRAP, chunk):
        idx = rng.integers(0, nb, size=(min(chunk, N_BOOTSTRAP - start), nb))
        c = cnt[idx].sum(axis=1)
        boot[start : start + idx.shape[0]] = f(s_eo[idx].sum(axis=1) / c) - f(s_lo[idx].sum(axis=1) / c)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "delta": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_width": float(hi - lo),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_blocks": nb,
    }


def conflict_slices(df: pd.DataFrame, col: str = "observable_conflict_nonref_v0", q=(0.75, 0.95), ref: pd.Series | None = None):
    ref = df[col] if ref is None else ref
    lo, hi = np.nanquantile(ref, q)
    k = df[col].to_numpy()
    return {"low": k <= lo, "mid": (k > lo) & (k <= hi), "high": k > hi}, (float(lo), float(hi))


def run_bootstrap() -> pd.DataFrame:
    full = load_year(MAIN_YEAR)
    df = robust_frame(MAIN_YEAR)
    y = df["avamet_agg_mm"].to_numpy(np.float64)
    loc = df[LOCAL_COL].to_numpy(np.float64)
    # conflict bins from the >=1 universe, as in the manuscript
    cbins, _ = conflict_slices(df, ref=full["observable_conflict_nonref_v0"])
    slices = {"all hours": np.ones(y.size, bool), "positive rain": y > 0, **{f"{k} conflict": v for k, v in cbins.items()}}
    blocks = {
        "hour": hour_blocks(df),
        "day": calendar_blocks(df, 1),
        "3-day": calendar_blocks(df, 3),
        "7-day": calendar_blocks(df, 7),
        "wet-spell event": spell_blocks(df),
    }
    rows = []
    for slice_name, mask in slices.items():
        for eo in EO_SOURCES:
            p = df[SOURCES[eo]].to_numpy(np.float64)
            for block_name, codes in blocks.items():
                res = bootstrap_delta(y[mask], p[mask], loc[mask], codes[mask])
                rows.append({"slice": slice_name, "source": eo, "block": block_name, **res})
    out = pd.DataFrame(rows)
    hour_w = out[out["block"] == "hour"].set_index(["slice", "source"])["ci_width"]
    out["width_ratio_vs_hour"] = out.apply(lambda r: r["ci_width"] / hour_w[(r["slice"], r["source"])], axis=1)
    out.to_csv(OUT_DIR / "bootstrap_block_sensitivity_2023.csv", index=False)
    spells = spell_table(df)
    spells.to_csv(OUT_DIR / "wet_spells_2023.csv")
    print(
        f"wet spells: n={len(spells)}, median length={spells['length_h'].median():.0f} h, "
        f"p90={spells['length_h'].quantile(0.9):.0f} h, max={spells['length_h'].max():.0f} h"
    )
    return out


# ---------------------------------------------------------------- conflict

def add_relative_conflict(df: pd.DataFrame) -> pd.DataFrame:
    mean_prod = df[["imerg_mm", "era5_mm", "euradclim_on_imerg_mm"]].mean(axis=1)
    df["conflict_rel"] = df["observable_conflict_nonref_v0"] / (mean_prod + REL_CONFLICT_EPS_MM)
    df["product_mean_mm"] = mean_prod
    return df


def delta_rows(df: pd.DataFrame, masks: dict[str, np.ndarray], extra: dict[str, object]) -> list[dict[str, object]]:
    y = df["avamet_agg_mm"].to_numpy(np.float64)
    loc = df[LOCAL_COL].to_numpy(np.float64)
    codes = hour_blocks(df)
    rows = []
    for name, mask in masks.items():
        if mask.sum() == 0:
            continue
        for eo in EO_SOURCES:
            p = df[SOURCES[eo]].to_numpy(np.float64)
            res = bootstrap_delta(y[mask], p[mask], loc[mask], codes[mask])
            rows.append(
                {
                    **extra,
                    "stratum": name,
                    "source": eo,
                    "n": int(mask.sum()),
                    "wet_share": float((y[mask] > 0).mean()),
                    "mean_target_mm": float(y[mask].mean()),
                    "mae_local": float(np.mean(np.abs(loc[mask] - y[mask]))),
                    **res,
                }
            )
    return rows


def run_conflict() -> None:
    full = add_relative_conflict(load_year(MAIN_YEAR))
    df = add_relative_conflict(robust_frame(MAIN_YEAR))
    y = df["avamet_agg_mm"].to_numpy(np.float64)
    pos = y > 0

    rows = []
    abs_bins, _ = conflict_slices(df, ref=full["observable_conflict_nonref_v0"])
    rows += delta_rows(df, abs_bins, {"index": "absolute kappa", "subset": "all hours"})
    rel_bins, _ = conflict_slices(df, col="conflict_rel", ref=full["conflict_rel"])
    rows += delta_rows(df, rel_bins, {"index": "relative kappa", "subset": "all hours"})
    # positive-rain only, bins re-defined on positive-rain rows
    dpos = df.loc[pos].reset_index(drop=True)
    for col, name in [("observable_conflict_nonref_v0", "absolute kappa"), ("conflict_rel", "relative kappa")]:
        b, _ = conflict_slices(dpos, col=col)
        rows += delta_rows(dpos, b, {"index": name, "subset": "positive rain (bins on wet rows)"})
    pd.DataFrame(rows).to_csv(OUT_DIR / "conflict_normalized_2023.csv", index=False)

    # conflict x intensity: within each positive-rain intensity bin, split by within-bin conflict tercile
    dpos["intensity_bin"] = pd.cut(dpos["avamet_agg_mm"], INTENSITY_BINS, labels=INTENSITY_LABELS)
    cross = []
    for ib in INTENSITY_LABELS:
        sub = dpos.loc[dpos["intensity_bin"] == ib].reset_index(drop=True)
        for col, name in [("observable_conflict_nonref_v0", "absolute kappa"), ("conflict_rel", "relative kappa")]:
            t1, t2 = np.quantile(sub[col], [1 / 3, 2 / 3])
            k = sub[col].to_numpy()
            masks = {"T1 (low)": k <= t1, "T2": (k > t1) & (k <= t2), "T3 (high)": k > t2}
            cross += delta_rows(sub, masks, {"index": name, "intensity_bin": ib})
    pd.DataFrame(cross).to_csv(OUT_DIR / "conflict_by_intensity_2023.csv", index=False)

    # Spearman diagnostics
    loc = df[LOCAL_COL].to_numpy(np.float64)
    diag = []
    for col, name in [("observable_conflict_nonref_v0", "absolute kappa"), ("conflict_rel", "relative kappa")]:
        k = df[col].to_numpy()
        diag.append({"index": name, "pair": "kappa vs target (wet rows)", "rho": spearmanr(k[pos], y[pos]).statistic})
        any_prod = df["product_mean_mm"].to_numpy() >= 0.5
        diag.append({"index": name, "pair": "kappa vs product mean (rows with product mean >= 0.5 mm)", "rho": spearmanr(k[any_prod], df["product_mean_mm"].to_numpy()[any_prod]).statistic})
        for eo in EO_SOURCES:
            p = df[SOURCES[eo]].to_numpy(np.float64)
            gap = np.abs(p - y) - np.abs(loc - y)
            diag.append({"index": name, "pair": f"kappa vs error gap {eo} (wet rows)", "rho": spearmanr(k[pos], gap[pos]).statistic})
    pd.DataFrame(diag).to_csv(OUT_DIR / "conflict_spearman_2023.csv", index=False)


# ---------------------------------------------------------------- thresholds

def run_thresholds() -> None:
    full = load_year(MAIN_YEAR)
    df = robust_frame(MAIN_YEAR)
    rows = []
    for q in [(0.70, 0.90), (0.75, 0.95), (0.80, 0.975), (0.90, 0.99)]:
        b, (lo, hi) = conflict_slices(df, q=q, ref=full["observable_conflict_nonref_v0"])
        rows += delta_rows(df, b, {"cutoffs": f"p{q[0]*100:g}/p{q[1]*100:g}", "kappa_lo_mm": lo, "kappa_hi_mm": hi})
    pd.DataFrame(rows).to_csv(OUT_DIR / "conflict_threshold_sensitivity_2023.csv", index=False)


# ---------------------------------------------------------------- years and extremes

def run_years() -> None:
    rows = []
    frames = {y: load_year(y) for y in YEARS}
    for year in YEARS:
        df = frames[year]
        df = df.loc[df["avamet_n_active"] >= 2].reset_index(drop=True)
        y = df["avamet_agg_mm"].to_numpy(np.float64)
        p99 = float(np.quantile(y[y > 0], 0.99))
        masks = {
            "all hours": np.ones(y.size, bool),
            "positive rain": y > 0,
            "target > 10 mm": y > 10,
            "target > 20 mm": y > 20,
            f"top 1% wet (> p99 of wet rows)": y > p99,
        }
        for r in delta_rows(df, masks, {"year": year, "wet_share_year": float((y > 0).mean()), "p99_wet_mm": p99}):
            rows.append(r)
    pd.DataFrame(rows).to_csv(OUT_DIR / "per_year_extremes.csv", index=False)

    # routing transfer per held-out year, incl. extremes
    pol_rows = []
    for train, test in zip(YEARS[:-1], YEARS[1:]):
        policy = ext.train_conflict_policy(frames[train])
        tdf = frames[test]
        pred, local_used = ext.build_policy_predictions(tdf, policy)
        robust = tdf["avamet_n_active"].to_numpy() >= 2
        y = tdf["avamet_agg_mm"].to_numpy(np.float64)[robust]
        pred = pred[robust]
        loc = tdf[LOCAL_COL].to_numpy(np.float64)[robust]
        eur = tdf["euradclim_on_imerg_mm"].to_numpy(np.float64)[robust]
        for name, m in {"all hours": np.ones(y.size, bool), "positive rain": y > 0, "target > 10 mm": y > 10}.items():
            pol_rows.append(
                {
                    "train": train,
                    "test": test,
                    "wet_share_train": float((tdf.shape[0] and (frames[train].loc[frames[train]["avamet_n_active"] >= 2, "avamet_agg_mm"] > 0).mean())),
                    "wet_share_test": float((y > 0).mean()),
                    "low_choice": policy["low_conflict_choice"],
                    "high_choice": policy["high_conflict_choice"],
                    "slice": name,
                    "n": int(m.sum()),
                    "local_usage": float(local_used[robust][m].mean()),
                    "mae_policy": float(np.mean(np.abs(pred[m] - y[m]))),
                    "mae_local": float(np.mean(np.abs(loc[m] - y[m]))),
                    "mae_euradclim": float(np.mean(np.abs(eur[m] - y[m]))),
                }
            )
    pd.DataFrame(pol_rows).to_csv(OUT_DIR / "routing_by_year_extremes.csv", index=False)


BLOCKS = {
    "metrics": run_metrics,
    "bootstrap": run_bootstrap,
    "conflict": run_conflict,
    "thresholds": run_thresholds,
    "years": run_years,
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or list(BLOCKS):
        print(f"== {name}", flush=True)
        BLOCKS[name]()


if __name__ == "__main__":
    main()
