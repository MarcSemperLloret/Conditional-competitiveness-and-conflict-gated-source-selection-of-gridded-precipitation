#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from build_delta_mae_by_support_score_and_regime_v1 import (
    CANONICAL_PARQUET,
    EO_COLUMNS,
    FIXED_LOCAL_BASELINE,
    LOCAL_BASELINES,
    STATIC_PARQUET,
    STATION_CSV,
    TIME_END,
    TIME_START,
    build_prediction_frame,
    compute_metrics,
)
from run_delta_mae_block_bootstrap_v1 import bootstrap_mae_difference, factorize_blocks


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper_11_density_thresholds" / "results" / "reviewer_hardening_v1"
DEFAULT_N_BOOTSTRAP = 1500
DEFAULT_SEED = 42
FIG_DPI = 400

BG = "#ffffff"
INK = "#1f2a33"
GRID = "#d9e1ea"
ZERO = "#aab5c0"
IMERG = "#1677b3"
ERA5 = "#e18c31"
EURAD = "#2f8d62"

EO_LABELS = {
    "source_imerg": "IMERG",
    "source_era5": "ERA5",
    "source_euradclim": "EURADCLIM",
}
LOCAL_LABELS = {
    "domain_idw_leavecell_knn08": "8-neighbor IDW",
    "domain_idw_leavecell_knn64": "64-neighbor IDW",
    "domain_idw_leavecell_knn64_r15km": "64-neighbor IDW within 15 km",
}
COLORS = {
    "IMERG": IMERG,
    "ERA5": ERA5,
    "EURADCLIM": EURAD,
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Aptos", "Segoe UI", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
        "savefig.dpi": FIG_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save_exports(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.png")
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    plt.close(fig)


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "null"
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_f):
        return "null"
    return f"{value_f:.{digits}f}"


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS[name],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=7.5,
            label=name,
        )
        for name in ["IMERG", "ERA5", "EURADCLIM"]
    ]


def build_positive_rain_conflict_bootstrap(
    df: pd.DataFrame,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    robust_mask = df["avamet_n_active"].to_numpy(dtype=np.int16, copy=False) >= 2
    positive_mask = df["avamet_agg_mm"].to_numpy(dtype=np.float32, copy=False) > 0.0
    conflict = df["observable_conflict_bin"].astype(str).to_numpy()
    y_true = df["avamet_agg_mm"].to_numpy(dtype=np.float32, copy=False)
    fixed_local = df[FIXED_LOCAL_BASELINE].to_numpy(dtype=np.float32, copy=False)
    block_codes, _ = factorize_blocks(df["time_utc"].to_numpy())

    slices = [
        ("overall_positive_target", "Overall positive-rain", robust_mask & positive_mask),
        ("positive_conflict_low", "Low conflict", robust_mask & positive_mask & (conflict == "low_le_p75")),
        ("positive_conflict_mid", "Mid conflict", robust_mask & positive_mask & (conflict == "mid_p75_p95")),
        ("positive_conflict_tail", "High conflict", robust_mask & positive_mask & (conflict == "tail_gt_p95")),
    ]

    rows: list[dict[str, object]] = []
    for order, (slice_id, slice_label, mask) in enumerate(slices):
        subset = df.loc[mask]
        for eo_name, eo_col in EO_COLUMNS.items():
            result = bootstrap_mae_difference(
                y_true=y_true[mask],
                pred_eo=df[eo_col].to_numpy(dtype=np.float32, copy=False)[mask],
                pred_local=fixed_local[mask],
                block_codes=block_codes[mask],
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            rows.append(
                {
                    "slice_id": slice_id,
                    "slice_label": slice_label,
                    "slice_order": order,
                    "eo_baseline": eo_name,
                    "eo_label": EO_LABELS[eo_name],
                    "local_baseline": FIXED_LOCAL_BASELINE,
                    "local_label": LOCAL_LABELS[FIXED_LOCAL_BASELINE],
                    "n_rows": int(mask.sum()),
                    "n_hours": int(subset["time_utc"].nunique()),
                    "mean_target_mm": float(subset["avamet_agg_mm"].mean()),
                    **result,
                }
            )

    out = pd.DataFrame(rows).sort_values(["slice_order", "eo_label"]).reset_index(drop=True)
    return out


def plot_positive_rain_conflict(bootstrap_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    products = ["IMERG", "ERA5", "EURADCLIM"]
    offsets = {"IMERG": -0.22, "ERA5": 0.0, "EURADCLIM": 0.22}
    slices = bootstrap_df["slice_label"].drop_duplicates().tolist()
    y_base = np.arange(len(slices))[::-1].astype(float)

    ax.axvline(0.0, color=ZERO, linewidth=1.2, linestyle="--", zorder=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.9)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Positive-rain penalties remain under all conflict regimes", loc="left", pad=12)
    ax.set_xlabel("Delta MAE vs fixed local benchmark (mm)")

    for product in products:
        sub = bootstrap_df[bootstrap_df["eo_label"] == product].copy()
        y_values: list[float] = []
        x_values: list[float] = []
        xerr_low: list[float] = []
        xerr_high: list[float] = []
        for idx, slice_name in enumerate(slices):
            row = sub[sub["slice_label"] == slice_name].iloc[0]
            y = y_base[idx] + offsets[product]
            delta = float(row["delta_mae_eo_minus_local"])
            y_values.append(y)
            x_values.append(delta)
            xerr_low.append(delta - float(row["ci_low"]))
            xerr_high.append(float(row["ci_high"]) - delta)
        ax.errorbar(
            x_values,
            y_values,
            xerr=np.vstack([xerr_low, xerr_high]),
            fmt="o",
            color=COLORS[product],
            ecolor=COLORS[product],
            elinewidth=1.8,
            capsize=3.5,
            markersize=7.0,
            markerfacecolor=COLORS[product],
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )

    ax.set_yticks(y_base)
    ax.set_yticklabels(slices)
    upper = float(bootstrap_df["ci_high"].max()) * 1.08
    ax.set_xlim(0.0, upper)
    ax.legend(handles=legend_handles(), loc="upper right", frameon=False, ncol=1)

    save_exports(fig, "figure_positive_rain_conflict_bootstrap_v1")


def build_benchmark_sensitivity(
    df: pd.DataFrame,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    robust_mask = df["avamet_n_active"].to_numpy(dtype=np.int16, copy=False) >= 2
    positive_mask = df["avamet_agg_mm"].to_numpy(dtype=np.float32, copy=False) > 0.0
    conflict = df["observable_conflict_bin"].astype(str).to_numpy()
    y_true = df["avamet_agg_mm"].to_numpy(dtype=np.float32, copy=False)
    block_codes, _ = factorize_blocks(df["time_utc"].to_numpy())

    slices = [
        ("overall_all", "Overall", robust_mask),
        ("overall_positive_target", "Positive-rain only", robust_mask & positive_mask),
        ("conflict_tail", "High-conflict tail", robust_mask & (conflict == "tail_gt_p95")),
    ]

    rows: list[dict[str, object]] = []
    for slice_order, (slice_id, slice_label, mask) in enumerate(slices):
        subset = df.loc[mask]
        for local_baseline in LOCAL_BASELINES:
            local_pred = df[local_baseline].to_numpy(dtype=np.float32, copy=False)
            local_metrics = compute_metrics(y_true[mask], local_pred[mask])
            for eo_name, eo_col in EO_COLUMNS.items():
                result = bootstrap_mae_difference(
                    y_true=y_true[mask],
                    pred_eo=df[eo_col].to_numpy(dtype=np.float32, copy=False)[mask],
                    pred_local=local_pred[mask],
                    block_codes=block_codes[mask],
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )
                rows.append(
                    {
                        "slice_id": slice_id,
                        "slice_label": slice_label,
                        "slice_order": slice_order,
                        "local_baseline": local_baseline,
                        "local_label": LOCAL_LABELS[local_baseline],
                        "eo_baseline": eo_name,
                        "eo_label": EO_LABELS[eo_name],
                        "n_rows": int(mask.sum()),
                        "n_hours": int(subset["time_utc"].nunique()),
                        "local_mae_mm": local_metrics["mae_mm"],
                        **result,
                    }
                )

    return pd.DataFrame(rows).sort_values(["slice_order", "local_label", "eo_label"]).reset_index(drop=True)


def build_domain_support_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    static_df = pd.read_parquet(STATIC_PARQUET).copy()
    static_df = static_df.loc[static_df["inside_cv_polygon"].fillna(False).astype(bool)].copy()
    station_df = pd.read_csv(STATION_CSV)
    station_df = station_df.loc[station_df["imerg_cell_in_cv_polygon"].fillna(False).astype(bool)].copy()

    con = duckdb.connect()
    query = f"""
        SELECT
            grid_id,
            COUNT(*) AS n_hours_total,
            SUM(
                CASE
                    WHEN avamet_n_active >= 1
                     AND isfinite(avamet_agg_mm)
                     AND isfinite(imerg_mm)
                     AND isfinite(era5_mm)
                     AND isfinite(euradclim_on_imerg_mm)
                    THEN 1 ELSE 0
                END
            ) AS common_eval_hours_ge1,
            SUM(
                CASE
                    WHEN avamet_n_active >= 2
                     AND isfinite(avamet_agg_mm)
                     AND isfinite(imerg_mm)
                     AND isfinite(era5_mm)
                     AND isfinite(euradclim_on_imerg_mm)
                    THEN 1 ELSE 0
                END
            ) AS common_eval_hours_ge2,
            AVG(
                CASE
                    WHEN avamet_n_active >= 1
                     AND isfinite(avamet_agg_mm)
                     AND isfinite(imerg_mm)
                     AND isfinite(era5_mm)
                     AND isfinite(euradclim_on_imerg_mm)
                    THEN CAST(avamet_n_active AS DOUBLE)
                    ELSE NULL
                END
            ) AS mean_n_active_common_eval,
            AVG(
                CASE
                    WHEN avamet_n_active >= 1
                     AND isfinite(avamet_agg_mm)
                     AND isfinite(imerg_mm)
                     AND isfinite(era5_mm)
                     AND isfinite(euradclim_on_imerg_mm)
                    THEN avamet_support_score_v0
                    ELSE NULL
                END
            ) AS mean_support_score_common_eval
        FROM read_parquet('{CANONICAL_PARQUET}')
        WHERE CAST(time_utc AS TIMESTAMP) >= ?
          AND CAST(time_utc AS TIMESTAMP) < ?
        GROUP BY grid_id
        ORDER BY grid_id
    """
    agg = con.execute(query, [TIME_START, TIME_END]).fetchdf()
    con.close()

    merged = static_df.merge(agg, on="grid_id", how="left")
    merged["common_eval_share_full_year"] = merged["common_eval_hours_ge1"] / merged["n_hours_total"]
    merged["robust_support_share_full_year"] = merged["common_eval_hours_ge2"] / merged["n_hours_total"]
    merged["robust_support_share_within_common_eval"] = np.where(
        merged["common_eval_hours_ge1"] > 0,
        merged["common_eval_hours_ge2"] / merged["common_eval_hours_ge1"],
        np.nan,
    )
    return merged, station_df


def _scatter_grid(
    ax: plt.Axes,
    cell_df: pd.DataFrame,
    value_col: str,
    cmap: str,
    title: str,
    cbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
    overlay_stations: pd.DataFrame | None = None,
) -> None:
    sc = ax.scatter(
        cell_df["lon"],
        cell_df["lat"],
        c=cell_df[value_col],
        cmap=cmap,
        s=190,
        marker="s",
        linewidths=0.0,
        vmin=vmin,
        vmax=vmax,
        alpha=0.95,
    )
    if overlay_stations is not None and not overlay_stations.empty:
        ax.scatter(
            overlay_stations["lon"],
            overlay_stations["lat"],
            s=9,
            color="#111111",
            alpha=0.55,
            linewidths=0.0,
            zorder=4,
        )

    ax.set_title(title, loc="left", pad=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color=GRID, linewidth=0.6, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(cbar_label)


def plot_domain_support(cell_df: pd.DataFrame, station_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.8))
    fig.subplots_adjust(top=0.86, wspace=0.34)
    _scatter_grid(
        axes[0],
        cell_df,
        value_col="avamet_cell_station_count_total",
        cmap="YlOrBr",
        title="A. Static gauge footprint",
        cbar_label="Total AVAMET stations mapped to cell",
        overlay_stations=station_df,
    )
    _scatter_grid(
        axes[1],
        cell_df,
        value_col="robust_support_share_full_year",
        cmap="viridis",
        title="B. Robust support in 2023",
        cbar_label="Share of 2023 hours with >=2 active gauges",
        vmin=0.0,
        vmax=max(0.20, float(cell_df["robust_support_share_full_year"].max())),
    )
    fig.suptitle("Domain and support heterogeneity", x=0.07, y=0.97, ha="left", fontsize=15, fontweight="bold")
    save_exports(fig, "figure_domain_support_overview_v1")


def build_summary_markdown(
    positive_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    cell_df: pd.DataFrame,
    station_df: pd.DataFrame,
) -> str:
    lines = [
        "# Reviewer Hardening Package v1",
        "",
        "- Purpose: low-cost additions that directly address reviewer concerns about conflict, benchmark fairness, and spatial support heterogeneity.",
        f"- Canonical parquet: `{CANONICAL_PARQUET}`",
        f"- Static grid parquet: `{STATIC_PARQUET}`",
        f"- Station inventory: `{STATION_CSV}`",
        f"- Bootstrap resamples: {DEFAULT_N_BOOTSTRAP}",
        "",
        "## Positive-rain conflict bootstrap",
    ]

    for _, row in positive_df.iterrows():
        lines.append(
            f"- `{row['slice_label']}` / `{row['eo_label']}`: delta MAE vs fixed local = "
            f"{fmt(row['delta_mae_eo_minus_local'])} "
            f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}], "
            f"n_rows = {int(row['n_rows'])}, n_hours = {int(row['n_hours'])}"
        )

    lines.extend(["", "## Fixed-benchmark sensitivity"])
    for slice_label in ["Overall", "Positive-rain only", "High-conflict tail"]:
        sub = sensitivity_df[sensitivity_df["slice_label"] == slice_label].copy()
        eo_order = ["IMERG", "ERA5", "EURADCLIM"]
        for eo_label in eo_order:
            sub_eo = sub[sub["eo_label"] == eo_label].copy()
            if sub_eo.empty:
                continue
            min_delta = float(sub_eo["delta_mae_eo_minus_local"].min())
            max_delta = float(sub_eo["delta_mae_eo_minus_local"].max())
            lines.append(
                f"- `{slice_label}` / `{eo_label}`: delta MAE stays in "
                f"[{fmt(min_delta)}, {fmt(max_delta)}] across the three fixed local baselines."
            )

    lines.extend(
        [
            "",
            "## Domain and support summary",
            f"- Cells inside CV polygon: {int(cell_df.shape[0])}",
            f"- Stations inside CV polygon: {int(station_df.shape[0])}",
            f"- Median stations per cell: {fmt(cell_df['avamet_cell_station_count_total'].median(), digits=1)}",
            f"- Cells with at least one common-evaluable hour: {int((cell_df['common_eval_hours_ge1'] > 0).sum())}",
            f"- Cells with at least one robust-support hour (>=2 active gauges): {int((cell_df['common_eval_hours_ge2'] > 0).sum())}",
            f"- Median share of 2023 hours with robust support: {fmt(cell_df['robust_support_share_full_year'].median())}",
            f"- Upper quartile share of 2023 hours with robust support: {fmt(cell_df['robust_support_share_full_year'].quantile(0.75))}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_prediction_frame()

    positive_df = build_positive_rain_conflict_bootstrap(df)
    sensitivity_df = build_benchmark_sensitivity(df)
    cell_df, station_df = build_domain_support_summary()

    positive_csv = OUT_DIR / "positive_rain_conflict_bootstrap_v1.csv"
    sensitivity_csv = OUT_DIR / "fixed_local_sensitivity_v1.csv"
    cell_csv = OUT_DIR / "domain_support_cell_summary_v1.csv"
    md_path = OUT_DIR / "reviewer_hardening_notes_v1.md"
    manifest_path = OUT_DIR / "reviewer_hardening_manifest_v1.json"

    positive_df.to_csv(positive_csv, index=False)
    sensitivity_df.to_csv(sensitivity_csv, index=False)
    cell_df.to_csv(cell_csv, index=False)

    plot_positive_rain_conflict(positive_df)
    plot_domain_support(cell_df, station_df)

    md_path.write_text(build_summary_markdown(positive_df, sensitivity_df, cell_df, station_df), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "outputs": {
                    "positive_rain_conflict_bootstrap_v1.csv": str(positive_csv),
                    "fixed_local_sensitivity_v1.csv": str(sensitivity_csv),
                    "domain_support_cell_summary_v1.csv": str(cell_csv),
                    "figure_positive_rain_conflict_bootstrap_v1.png": str(OUT_DIR / "figure_positive_rain_conflict_bootstrap_v1.png"),
                    "figure_positive_rain_conflict_bootstrap_v1.pdf": str(OUT_DIR / "figure_positive_rain_conflict_bootstrap_v1.pdf"),
                    "figure_domain_support_overview_v1.png": str(OUT_DIR / "figure_domain_support_overview_v1.png"),
                    "figure_domain_support_overview_v1.pdf": str(OUT_DIR / "figure_domain_support_overview_v1.pdf"),
                    "reviewer_hardening_notes_v1.md": str(md_path),
                },
                "bootstrap_resamples": DEFAULT_N_BOOTSTRAP,
                "seed": DEFAULT_SEED,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
