#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import build_delta_mae_by_support_score_and_regime_v1 as base
from run_delta_mae_block_bootstrap_v1 import bootstrap_mae_difference, factorize_blocks


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "temporal_sensitivity_v1"
DEFAULT_N_BOOTSTRAP = 1500
DEFAULT_SEED = 42

EO_LABELS = {
    "source_imerg": "IMERG",
    "source_era5": "ERA5",
    "source_euradclim": "EURADCLIM",
}


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "null"
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{value_f:.{digits}f}"


def build_year_frame(year: int) -> pd.DataFrame:
    base.TIME_START = pd.Timestamp(f"{year}-01-01 00:00:00")
    base.TIME_END = pd.Timestamp(f"{year + 1}-01-01 00:00:00")
    return base.build_prediction_frame()


def build_year_summary(df: pd.DataFrame, year: int) -> pd.DataFrame:
    robust = df["avamet_n_active"].to_numpy(dtype="int16", copy=False) >= 2
    positive = df["avamet_agg_mm"].to_numpy(dtype="float32", copy=False) > 0.0
    conflict = df["observable_conflict_bin"].astype(str).to_numpy()
    y_true = df["avamet_agg_mm"].to_numpy(dtype="float32", copy=False)
    local = df[base.FIXED_LOCAL_BASELINE].to_numpy(dtype="float32", copy=False)
    block_codes, _ = factorize_blocks(df["time_utc"].to_numpy())

    slices = [
        ("overall_all", "Overall", robust),
        ("overall_positive_target", "Positive-rain only", robust & positive),
        ("conflict_low", "Low conflict", robust & (conflict == "low_le_p75")),
        ("conflict_mid", "Mid conflict", robust & (conflict == "mid_p75_p95")),
        ("conflict_tail", "High conflict", robust & (conflict == "tail_gt_p95")),
    ]

    rows: list[dict[str, object]] = []
    for slice_order, (slice_id, slice_label, mask) in enumerate(slices):
        subset = df.loc[mask]
        for eo_name, eo_col in base.EO_COLUMNS.items():
            result = bootstrap_mae_difference(
                y_true=y_true[mask],
                pred_eo=df[eo_col].to_numpy(dtype="float32", copy=False)[mask],
                pred_local=local[mask],
                block_codes=block_codes[mask],
                n_bootstrap=DEFAULT_N_BOOTSTRAP,
                seed=DEFAULT_SEED,
            )
            rows.append(
                {
                    "year": year,
                    "slice_id": slice_id,
                    "slice_label": slice_label,
                    "slice_order": slice_order,
                    "eo_baseline": eo_name,
                    "eo_label": EO_LABELS[eo_name],
                    "n_rows": int(mask.sum()),
                    "n_hours": int(subset["time_utc"].nunique()),
                    "wet_share": float((subset["avamet_agg_mm"] > 0.0).mean()),
                    **result,
                }
            )
    return pd.DataFrame(rows).sort_values(["slice_order", "eo_label"]).reset_index(drop=True)


def build_markdown(summary_df: pd.DataFrame) -> str:
    lines = [
        "# Temporal Sensitivity v1",
        "",
        "- Purpose: compare the main inferential year (2023) against a same-protocol sensitivity year (2022).",
        f"- Fixed local comparator: `{base.FIXED_LOCAL_BASELINE}`",
        f"- Bootstrap resamples: {DEFAULT_N_BOOTSTRAP}",
        "",
        "## Main comparison",
    ]
    for slice_label in ["Overall", "Positive-rain only", "Low conflict", "Mid conflict", "High conflict"]:
        subset = summary_df[summary_df["slice_label"] == slice_label].copy()
        if subset.empty:
            continue
        lines.append(f"- `{slice_label}`")
        for _, row in subset.iterrows():
            lines.append(
                f"  - {int(row['year'])} / {row['eo_label']}: "
                f"Î”MAE = {fmt(row['delta_mae_eo_minus_local'])} "
                f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}], "
                f"n_rows = {int(row['n_rows'])}, wet_share = {fmt(row['wet_share'], 6)}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in [2022, 2023]:
        df = build_year_frame(year)
        frames.append(build_year_summary(df, year))
    summary_df = pd.concat(frames, ignore_index=True)

    csv_path = OUT_DIR / "temporal_year_comparison_bootstrap_v1.csv"
    md_path = OUT_DIR / "temporal_year_comparison_bootstrap_v1.md"
    manifest_path = OUT_DIR / "temporal_year_comparison_manifest_v1.json"

    summary_df.to_csv(csv_path, index=False)
    md_path.write_text(build_markdown(summary_df), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "years": [2022, 2023],
                "fixed_local_baseline": base.FIXED_LOCAL_BASELINE,
                "bootstrap_resamples": DEFAULT_N_BOOTSTRAP,
                "seed": DEFAULT_SEED,
                "output_csv": str(csv_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

