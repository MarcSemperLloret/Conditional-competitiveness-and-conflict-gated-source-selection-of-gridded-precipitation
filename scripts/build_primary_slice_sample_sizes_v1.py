#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from build_delta_mae_by_support_score_and_regime_v1 import build_prediction_frame


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results"


def build_slice_table(df: pd.DataFrame) -> pd.DataFrame:
    robust_mask = df["avamet_n_active"] >= 2
    conflict = df["observable_conflict_bin"].astype(str)
    intensity = df["target_intensity_bin_v2"].astype(str)

    slices: list[tuple[str, str, pd.Series]] = [
        ("overall", "all", robust_mask),
        ("overall", "positive_target_only", robust_mask & (df["avamet_agg_mm"] > 0.0)),
        ("observable_conflict", "low_le_p75", robust_mask & (conflict == "low_le_p75")),
        ("observable_conflict", "mid_p75_p95", robust_mask & (conflict == "mid_p75_p95")),
        ("observable_conflict", "tail_gt_p95", robust_mask & (conflict == "tail_gt_p95")),
        ("target_intensity", "(0.1,1]", robust_mask & (intensity == "(0.1,1]")),
        ("target_intensity", "(1,5]", robust_mask & (intensity == "(1,5]")),
        ("target_intensity", "(5,10]", robust_mask & (intensity == "(5,10]")),
        ("target_intensity", "(10,20]", robust_mask & (intensity == "(10,20]")),
        ("target_intensity", ">20", robust_mask & (intensity == ">20")),
    ]

    rows: list[dict[str, object]] = []
    for slice_family, slice_label, mask in slices:
        subset = df.loc[mask].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "scenario": "robust_ge_2",
                "slice_family": slice_family,
                "slice_label": slice_label,
                "n_rows": int(subset.shape[0]),
                "n_hours": int(subset["time_utc"].nunique()),
                "n_positive_target": int((subset["avamet_agg_mm"] > 0.0).sum()),
                "wet_share": float((subset["avamet_agg_mm"] > 0.0).mean()),
                "mean_target_mm": float(subset["avamet_agg_mm"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_markdown(table_df: pd.DataFrame) -> str:
    lines = [
        "# Primary Slice Sample Sizes v1",
        "",
        "| Scenario | Slice family | Slice label | n_rows | n_hours | n_positive_target | wet_share | mean_target_mm |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in table_df.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['slice_family']} | {row['slice_label']} | "
            f"{int(row['n_rows'])} | {int(row['n_hours'])} | {int(row['n_positive_target'])} | "
            f"{float(row['wet_share']):.6f} | {float(row['mean_target_mm']):.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_prediction_frame()
    table_df = build_slice_table(df)

    csv_path = OUTPUT_DIR / "primary_slice_sample_sizes_v1.csv"
    md_path = OUTPUT_DIR / "primary_slice_sample_sizes_v1.md"
    json_path = OUTPUT_DIR / "primary_slice_sample_sizes_v1.json"

    table_df.to_csv(csv_path, index=False)
    md_path.write_text(build_markdown(table_df), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "scenario": "robust_ge_2",
                "n_rows": int(table_df.shape[0]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

