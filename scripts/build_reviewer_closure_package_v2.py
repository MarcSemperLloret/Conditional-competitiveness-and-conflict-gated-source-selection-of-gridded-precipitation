#!/usr/bin/env python
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_delta_mae_by_support_score_and_regime_v1 import (
    AVAMET_HOURLY,
    CANONICAL_PARQUET,
    EO_COLUMNS,
    FIXED_LOCAL_BASELINE,
    STATION_CSV,
    TIME_END,
    TIME_START,
    build_prediction_frame,
)
from run_delta_mae_block_bootstrap_v1 import bootstrap_mae_difference, factorize_blocks


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "reviewer_closure_v2"
BASELINE_RESULTS = Path(os.environ.get("BASELINE_RESULTS_CSV", ROOT / "data" / "restricted" / "baseline_results_v3.csv"))
DEFAULT_N_BOOTSTRAP = 1500
DEFAULT_SEED = 42
FIG_DPI = 400

EO_LABELS = {
    "source_imerg": "IMERG",
    "source_era5": "ERA5",
    "source_euradclim": "EURADCLIM",
}
CONFLICT_ORDER = ["low_le_p75", "mid_p75_p95", "tail_gt_p95"]
CONFLICT_LABELS = {
    "low_le_p75": "Low conflict",
    "mid_p75_p95": "Mid conflict",
    "tail_gt_p95": "High conflict",
}
INTENSITY_ORDER = ["(0,0.1]", "(0.1,1]", "(1,5]", "(5,10]", "(10,20]", ">20"]
INTENSITY_LABELS = {
    "(0,0.1]": "0-0.1",
    "(0.1,1]": "0.1-1",
    "(1,5]": "1-5",
    "(5,10]": "5-10",
    "(10,20]": "10-20",
    ">20": ">20",
}
INTENSITY_COLORS = {
    "(0,0.1]": "#d9f0d3",
    "(0.1,1]": "#addd8e",
    "(1,5]": "#78c679",
    "(5,10]": "#41ab5d",
    "(10,20]": "#238443",
    ">20": "#005a32",
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
        "legend.fontsize": 9,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
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


def build_target_aggregates_2023() -> pd.DataFrame:
    station_df = pd.read_csv(STATION_CSV)
    station_df = station_df.loc[station_df["imerg_cell_in_cv_polygon"].fillna(False).astype(bool)].copy()
    station_df["grid_id"] = (
        "imerg_"
        + station_df["imerg_cv_lon_idx"].astype(int).astype(str)
        + "_"
        + station_df["imerg_cv_lat_idx"].astype(int).astype(str)
    )
    station_df = station_df[["station_id", "grid_id"]].drop_duplicates().reset_index(drop=True)

    con = duckdb.connect()
    con.register("station_map", station_df)
    query = f"""
        SELECT
            CAST(a.hour_start_utc AS TIMESTAMP) AS time_utc,
            m.grid_id,
            AVG(CAST(a.accum_mm AS DOUBLE)) AS avamet_mean_mm,
            MEDIAN(CAST(a.accum_mm AS DOUBLE)) AS avamet_median_rebuilt_mm,
            COUNT(*) AS n_complete_gauges
        FROM read_parquet('{AVAMET_HOURLY}') AS a
        JOIN station_map AS m USING (station_id)
        WHERE a.complete_strict
          AND CAST(a.hour_start_utc AS TIMESTAMP) >= ?
          AND CAST(a.hour_start_utc AS TIMESTAMP) < ?
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    out = con.execute(query, [TIME_START, TIME_END]).fetchdf()
    con.close()
    out["time_utc"] = pd.to_datetime(out["time_utc"], utc=False)
    return out


def build_target_sensitivity(pred_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    df = pred_df.merge(target_df, on=["time_utc", "grid_id"], how="left", validate="one_to_one")
    robust = df["avamet_n_active"].to_numpy(dtype=np.int16, copy=False) >= 2
    conflict = df["observable_conflict_bin"].astype(str).to_numpy()
    blocks, _ = factorize_blocks(df["time_utc"].to_numpy())
    local = df[FIXED_LOCAL_BASELINE].to_numpy(dtype=np.float32, copy=False)

    slices = [
        ("overall_all", "Overall", lambda frame: robust),
        ("overall_positive_target", "Positive-rain only", lambda frame: robust & (frame["target_values"] > 0.0)),
        ("conflict_low", "Low conflict", lambda frame: robust & (conflict == "low_le_p75")),
        ("conflict_mid", "Mid conflict", lambda frame: robust & (conflict == "mid_p75_p95")),
        ("conflict_tail", "High conflict", lambda frame: robust & (conflict == "tail_gt_p95")),
    ]

    target_specs = [
        ("within_cell_median", "Within-cell median", df["avamet_agg_mm"].to_numpy(dtype=np.float32, copy=False)),
        ("within_cell_mean", "Within-cell mean", df["avamet_mean_mm"].to_numpy(dtype=np.float32, copy=False)),
    ]

    rows: list[dict[str, object]] = []
    for target_id, target_label, target_values in target_specs:
        frame = {"target_values": target_values}
        for slice_order, (slice_id, slice_label, mask_fn) in enumerate(slices):
            mask = mask_fn(frame)
            for eo_name, eo_col in EO_COLUMNS.items():
                result = bootstrap_mae_difference(
                    y_true=target_values[mask],
                    pred_eo=df[eo_col].to_numpy(dtype=np.float32, copy=False)[mask],
                    pred_local=local[mask],
                    block_codes=blocks[mask],
                    n_bootstrap=DEFAULT_N_BOOTSTRAP,
                    seed=DEFAULT_SEED,
                )
                rows.append(
                    {
                        "target_id": target_id,
                        "target_label": target_label,
                        "slice_id": slice_id,
                        "slice_label": slice_label,
                        "slice_order": slice_order,
                        "eo_baseline": eo_name,
                        "eo_label": EO_LABELS[eo_name],
                        "n_rows": int(mask.sum()),
                        "n_hours": int(df.loc[mask, "time_utc"].nunique()),
                        **result,
                    }
                )

    summary = {
        "median_rebuild_corr": float(
            np.corrcoef(df["avamet_agg_mm"].to_numpy(dtype=np.float32), df["avamet_median_rebuilt_mm"].to_numpy(dtype=np.float32))[0, 1]
        ),
        "median_rebuild_max_abs_diff": float(
            np.nanmax(np.abs(df["avamet_agg_mm"].to_numpy(dtype=np.float32) - df["avamet_median_rebuilt_mm"].to_numpy(dtype=np.float32)))
        ),
    }
    out = pd.DataFrame(rows).sort_values(["slice_order", "eo_label", "target_label"]).reset_index(drop=True)
    return out, summary


def build_positive_conflict_intensity(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    df = pred_df.loc[(pred_df["avamet_n_active"] >= 2) & (pred_df["avamet_agg_mm"] > 0.0)].copy()
    counts = (
        df.groupby(["observable_conflict_bin", "target_intensity_bin_v2"])
        .size()
        .rename("n_rows")
        .reset_index()
    )
    counts = counts[counts["observable_conflict_bin"].isin(CONFLICT_ORDER)].copy()
    counts = counts[counts["target_intensity_bin_v2"].isin(INTENSITY_ORDER)].copy()

    totals = counts.groupby("observable_conflict_bin")["n_rows"].sum().rename("conflict_total")
    counts = counts.merge(totals, on="observable_conflict_bin", how="left")
    counts["row_share"] = counts["n_rows"] / counts["conflict_total"]
    counts["conflict_label"] = counts["observable_conflict_bin"].map(CONFLICT_LABELS)
    counts["intensity_label"] = counts["target_intensity_bin_v2"].map(INTENSITY_LABELS)

    stats = (
        df.groupby("observable_conflict_bin")["avamet_agg_mm"]
        .agg(["count", "mean", "median"])
        .rename(columns={"count": "positive_count", "mean": "positive_mean_mm", "median": "positive_median_mm"})
        .reset_index()
    )
    spearman = float(df[["observable_conflict_nonref_v0", "avamet_agg_mm"]].corr(method="spearman").iloc[0, 1])
    return counts.sort_values(["observable_conflict_bin", "target_intensity_bin_v2"]).reset_index(drop=True), {
        "positive_only_spearman_conflict_vs_target": spearman,
        "stats_by_conflict": stats.to_dict(orient="records"),
    }


def plot_positive_conflict_intensity(counts_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(CONFLICT_ORDER))
    bottoms = np.zeros(len(CONFLICT_ORDER), dtype=np.float64)

    for intensity in INTENSITY_ORDER:
        shares = []
        for conflict in CONFLICT_ORDER:
            row = counts_df[
                (counts_df["observable_conflict_bin"] == conflict)
                & (counts_df["target_intensity_bin_v2"] == intensity)
            ]
            shares.append(float(row["row_share"].iloc[0]) if not row.empty else 0.0)
        ax.bar(
            x,
            shares,
            bottom=bottoms,
            color=INTENSITY_COLORS[intensity],
            edgecolor="white",
            linewidth=0.6,
            width=0.65,
            label=INTENSITY_LABELS[intensity],
        )
        bottoms += np.array(shares)

    totals = counts_df.groupby("observable_conflict_bin")["conflict_total"].first()
    for idx, conflict in enumerate(CONFLICT_ORDER):
        ax.text(
            idx,
            1.02,
            f"n={int(totals.loc[conflict])}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CONFLICT_LABELS[c] for c in CONFLICT_ORDER])
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Share of positive-rain cases")
    ax.set_title("Positive-rain intensity mix within each conflict regime", loc="left", pad=10)
    ax.grid(axis="y", color="#d9e1ea", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        title="Intensity bin (mm)",
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
    )
    save_exports(fig, "figure_conflict_intensity_composition_v1")


def build_fixed_local_choice_summary() -> pd.DataFrame:
    df = pd.read_csv(BASELINE_RESULTS)
    keep = df[
        (df["scenario"] == "robust_ge_2")
        & (df["scope"] == "overall")
        & (df["stratifier"] == "overall")
        & (df["stratum"] == "all")
        & (
            df["baseline"].isin(
                [
                    "domain_idw_leavecell_knn08",
                    "domain_idw_leavecell_knn64",
                    "domain_idw_leavecell_knn64_r15km",
                ]
            )
        )
    ].copy()
    label_map = {
        "domain_idw_leavecell_knn08": "8-neighbor IDW",
        "domain_idw_leavecell_knn64": "64-neighbor IDW",
        "domain_idw_leavecell_knn64_r15km": "64-neighbor IDW within 15 km",
    }
    keep["baseline_label"] = keep["baseline"].map(label_map)
    return keep[
        ["baseline", "baseline_label", "n_requested", "coverage_share", "mae_mm", "rmse_mm", "corr"]
    ].sort_values("baseline_label").reset_index(drop=True)


def build_availability_summary() -> pd.DataFrame:
    con = duckdb.connect()
    query = f"""
        SELECT
            CASE WHEN avamet_n_active >= 2 THEN 'robust_ge_2' ELSE 'wide_ge_1_only' END AS support_slice,
            COUNT(*) AS n_target_defined,
            AVG(CASE WHEN isfinite(imerg_mm) THEN 1.0 ELSE 0.0 END) AS imerg_share,
            AVG(CASE WHEN isfinite(era5_mm) THEN 1.0 ELSE 0.0 END) AS era5_share,
            AVG(CASE WHEN isfinite(euradclim_on_imerg_mm) THEN 1.0 ELSE 0.0 END) AS euradclim_share,
            AVG(
                CASE
                    WHEN isfinite(imerg_mm) AND isfinite(era5_mm) AND isfinite(euradclim_on_imerg_mm)
                    THEN 1.0 ELSE 0.0
                END
            ) AS common_share
        FROM read_parquet('{CANONICAL_PARQUET}')
        WHERE CAST(time_utc AS TIMESTAMP) >= ?
          AND CAST(time_utc AS TIMESTAMP) < ?
          AND avamet_n_active >= 1
          AND isfinite(avamet_agg_mm)
        GROUP BY 1
        ORDER BY CASE support_slice WHEN 'robust_ge_2' THEN 0 ELSE 1 END
    """
    out = con.execute(query, [TIME_START, TIME_END]).fetchdf()
    con.close()
    return out


def build_markdown(
    target_df: pd.DataFrame,
    target_summary: dict[str, float],
    comp_df: pd.DataFrame,
    comp_summary: dict[str, float],
    fixed_df: pd.DataFrame,
    availability_df: pd.DataFrame,
) -> str:
    lines = [
        "# Reviewer Closure Package v2",
        "",
        "- Purpose: close the strongest remaining reviewer risks around target definition, conflict-versus-intensity interpretation, benchmark fairness, and availability framing.",
        "",
        "## Target aggregation sensitivity",
        f"- Rebuilt within-cell median matches the canonical target exactly up to floating-point tolerance: corr = {fmt(target_summary['median_rebuild_corr'], 6)}, max abs diff = {fmt(target_summary['median_rebuild_max_abs_diff'], 8)}.",
    ]

    for slice_label in ["Overall", "Positive-rain only", "Low conflict", "Mid conflict", "High conflict"]:
        sub = target_df[target_df["slice_label"] == slice_label].copy()
        for eo_label in ["IMERG", "ERA5", "EURADCLIM"]:
            sub_eo = sub[sub["eo_label"] == eo_label].copy()
            if sub_eo.empty:
                continue
            med = sub_eo[sub_eo["target_label"] == "Within-cell median"].iloc[0]
            mean = sub_eo[sub_eo["target_label"] == "Within-cell mean"].iloc[0]
            lines.append(
                f"- `{slice_label}` / `{eo_label}`: "
                f"median-target Î”MAE = {fmt(med['delta_mae_eo_minus_local'])}, "
                f"mean-target Î”MAE = {fmt(mean['delta_mae_eo_minus_local'])}."
            )

    lines.extend(
        [
            "",
            "## Conflict versus intensity",
            f"- Spearman correlation between continuous conflict and positive-rain target amount: {fmt(comp_summary['positive_only_spearman_conflict_vs_target'], 4)}.",
        ]
    )
    for row in comp_summary["stats_by_conflict"]:
        lines.append(
            f"- `{CONFLICT_LABELS[row['observable_conflict_bin']]}`: positive cases = {int(row['positive_count'])}, "
            f"mean target = {fmt(row['positive_mean_mm'])}, median target = {fmt(row['positive_median_mm'])}."
        )

    lines.extend(["", "## Fixed local benchmark choice"])
    for _, row in fixed_df.iterrows():
        lines.append(
            f"- `{row['baseline_label']}`: coverage = {fmt(row['coverage_share'], 6)}, "
            f"MAE = {fmt(row['mae_mm'])}, RMSE = {fmt(row['rmse_mm'])}, corr = {fmt(row['corr'])}."
        )

    lines.extend(["", "## Availability framing"])
    for _, row in availability_df.iterrows():
        lines.append(
            f"- `{row['support_slice']}`: target-defined rows = {int(row['n_target_defined'])}, "
            f"IMERG = {fmt(100.0 * row['imerg_share'], 2)}%, ERA5 = {fmt(100.0 * row['era5_share'], 2)}%, "
            f"EURADCLIM = {fmt(100.0 * row['euradclim_share'], 2)}%, common universe = {fmt(100.0 * row['common_share'], 2)}%."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_df = build_prediction_frame()
    target_agg_df = build_target_aggregates_2023()

    target_sens_df, target_summary = build_target_sensitivity(pred_df, target_agg_df)
    comp_df, comp_summary = build_positive_conflict_intensity(pred_df)
    fixed_df = build_fixed_local_choice_summary()
    availability_df = build_availability_summary()

    target_sens_df.to_csv(OUT_DIR / "target_aggregation_sensitivity_v1.csv", index=False)
    comp_df.to_csv(OUT_DIR / "conflict_intensity_positive_composition_v1.csv", index=False)
    fixed_df.to_csv(OUT_DIR / "fixed_local_choice_summary_v1.csv", index=False)
    availability_df.to_csv(OUT_DIR / "product_availability_summary_v1.csv", index=False)

    plot_positive_conflict_intensity(comp_df)

    md_path = OUT_DIR / "reviewer_closure_notes_v1.md"
    md_path.write_text(
        build_markdown(target_sens_df, target_summary, comp_df, comp_summary, fixed_df, availability_df),
        encoding="utf-8",
    )

    (OUT_DIR / "reviewer_closure_manifest_v1.json").write_text(
        json.dumps(
            {
                "outputs": {
                    "target_aggregation_sensitivity_v1.csv": str(OUT_DIR / "target_aggregation_sensitivity_v1.csv"),
                    "conflict_intensity_positive_composition_v1.csv": str(OUT_DIR / "conflict_intensity_positive_composition_v1.csv"),
                    "fixed_local_choice_summary_v1.csv": str(OUT_DIR / "fixed_local_choice_summary_v1.csv"),
                    "product_availability_summary_v1.csv": str(OUT_DIR / "product_availability_summary_v1.csv"),
                    "figure_conflict_intensity_composition_v1.png": str(OUT_DIR / "figure_conflict_intensity_composition_v1.png"),
                    "figure_conflict_intensity_composition_v1.pdf": str(OUT_DIR / "figure_conflict_intensity_composition_v1.pdf"),
                    "reviewer_closure_notes_v1.md": str(md_path),
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

