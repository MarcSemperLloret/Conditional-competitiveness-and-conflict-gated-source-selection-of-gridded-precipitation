#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_delta_mae_by_support_score_and_regime_v1 import (
    EO_COLUMNS,
    FIXED_LOCAL_BASELINE,
    build_prediction_frame,
)
from run_delta_mae_block_bootstrap_v1 import bootstrap_mae_difference, factorize_blocks


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "bootstrap_block_sensitivity_v1"
DEFAULT_N_BOOTSTRAP = 1500
DEFAULT_SEED = 42

EO_LABELS = {
    "source_imerg": "IMERG",
    "source_era5": "ERA5",
    "source_euradclim": "EURADCLIM",
}
SLICE_ORDER = [
    ("overall_all", "Overall"),
    ("overall_positive_target", "Positive-rain only"),
    ("conflict_low", "Low conflict"),
    ("conflict_mid", "Mid conflict"),
    ("conflict_tail", "High conflict"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare hour-block and day-block bootstrap intervals for paper_11 inferential slices."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


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


def build_slice_masks(df: pd.DataFrame) -> list[dict[str, object]]:
    robust_mask = df["avamet_n_active"].to_numpy(dtype=np.int16, copy=False) >= 2
    positive_mask = df["avamet_agg_mm"].to_numpy(dtype=np.float32, copy=False) > 0.0
    conflict = df["observable_conflict_bin"].astype(str).to_numpy()
    return [
        {"slice_id": "overall_all", "slice_label": "Overall", "mask": robust_mask},
        {"slice_id": "overall_positive_target", "slice_label": "Positive-rain only", "mask": robust_mask & positive_mask},
        {"slice_id": "conflict_low", "slice_label": "Low conflict", "mask": robust_mask & (conflict == "low_le_p75")},
        {"slice_id": "conflict_mid", "slice_label": "Mid conflict", "mask": robust_mask & (conflict == "mid_p75_p95")},
        {"slice_id": "conflict_tail", "slice_label": "High conflict", "mask": robust_mask & (conflict == "tail_gt_p95")},
    ]


def build_long_summary(df: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    y_true = df["avamet_agg_mm"].to_numpy(dtype=np.float32, copy=False)
    fixed_local = df[FIXED_LOCAL_BASELINE].to_numpy(dtype=np.float32, copy=False)
    block_codes = {
        unit: factorize_blocks(df["time_utc"].to_numpy(), unit=unit)[0]
        for unit in ["hour", "day"]
    }

    rows: list[dict[str, object]] = []
    for slice_meta in build_slice_masks(df):
        slice_mask = slice_meta["mask"]
        if not np.any(slice_mask):
            continue
        for block_unit, codes in block_codes.items():
            for eo_name, eo_col in EO_COLUMNS.items():
                result = bootstrap_mae_difference(
                    y_true=y_true[slice_mask],
                    pred_eo=df[eo_col].to_numpy(dtype=np.float32, copy=False)[slice_mask],
                    pred_local=fixed_local[slice_mask],
                    block_codes=codes[slice_mask],
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )
                rows.append(
                    {
                        "slice_id": slice_meta["slice_id"],
                        "slice_label": slice_meta["slice_label"],
                        "eo_baseline": eo_name,
                        "eo_label": EO_LABELS[eo_name],
                        "block_unit": block_unit,
                        **result,
                    }
                )
    out = pd.DataFrame(rows)
    out["slice_order"] = out["slice_id"].map({slice_id: idx for idx, (slice_id, _) in enumerate(SLICE_ORDER)})
    out["eo_order"] = out["eo_label"].map({"IMERG": 0, "ERA5": 1, "EURADCLIM": 2})
    return out.sort_values(["slice_order", "eo_order", "block_unit"]).reset_index(drop=True)


def build_wide_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    hour_df = (
        long_df[long_df["block_unit"] == "hour"]
        .drop(columns=["block_unit"])
        .rename(columns=lambda col: f"hour_{col}" if col not in {"slice_id", "slice_label", "eo_baseline", "eo_label"} else col)
    )
    day_df = (
        long_df[long_df["block_unit"] == "day"]
        .drop(columns=["block_unit"])
        .rename(columns=lambda col: f"day_{col}" if col not in {"slice_id", "slice_label", "eo_baseline", "eo_label"} else col)
    )
    wide_df = hour_df.merge(day_df, on=["slice_id", "slice_label", "eo_baseline", "eo_label"], how="inner", validate="one_to_one")
    wide_df["hour_ci"] = wide_df.apply(
        lambda row: f"[{fmt(row['hour_ci_low'])}, {fmt(row['hour_ci_high'])}]",
        axis=1,
    )
    wide_df["day_ci"] = wide_df.apply(
        lambda row: f"[{fmt(row['day_ci_low'])}, {fmt(row['day_ci_high'])}]",
        axis=1,
    )
    wide_df["sign_match"] = np.sign(wide_df["hour_delta_mae_eo_minus_local"]) == np.sign(wide_df["day_delta_mae_eo_minus_local"])
    wide_df["ci_exclusion_match"] = wide_df["hour_ci_excludes_zero"] == wide_df["day_ci_excludes_zero"]
    return wide_df.sort_values(["hour_slice_order", "hour_eo_order"]).reset_index(drop=True)


def build_notes(wide_df: pd.DataFrame, n_bootstrap: int) -> str:
    sign_match = int(wide_df["sign_match"].sum())
    ci_match = int(wide_df["ci_exclusion_match"].sum())
    lines = [
        "# Block Bootstrap Sensitivity v1",
        "",
        "- Purpose: compare the manuscript's active hour-block bootstrap against a stricter day-block bootstrap on the main inferential slices.",
        f"- Resamples per design: {n_bootstrap}",
        f"- Sign agreement between hour- and day-block estimates: {sign_match}/{wide_df.shape[0]} rows.",
        f"- CI exclusion-of-zero agreement: {ci_match}/{wide_df.shape[0]} rows.",
        "",
        "| Slice | EO | Hour-block delta MAE [95% CI] | Day-block delta MAE [95% CI] |",
        "|------|----|---------------------------|--------------------------|",
    ]
    for _, row in wide_df.iterrows():
        lines.append(
            f"| {row['slice_label']} | {row['eo_label']} | "
            f"{fmt(row['hour_delta_mae_eo_minus_local'])} {row['hour_ci']} | "
            f"{fmt(row['day_delta_mae_eo_minus_local'])} {row['day_ci']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "long_csv": args.output_dir / "block_bootstrap_sensitivity_long_v1.csv",
        "wide_csv": args.output_dir / "block_bootstrap_sensitivity_wide_v1.csv",
        "notes_md": args.output_dir / "block_bootstrap_sensitivity_notes_v1.md",
        "manifest_json": args.output_dir / "block_bootstrap_sensitivity_manifest_v1.json",
    }
    if not args.overwrite:
        existing = [path for path in outputs.values() if path.exists()]
        if existing:
            raise FileExistsError(f"Outputs already exist. Re-run with --overwrite to replace them: {existing}")

    df = build_prediction_frame()
    long_df = build_long_summary(df, n_bootstrap=args.n_bootstrap, seed=args.seed)
    wide_df = build_wide_summary(long_df)

    long_df.to_csv(outputs["long_csv"], index=False)
    wide_df.to_csv(outputs["wide_csv"], index=False)
    outputs["notes_md"].write_text(build_notes(wide_df, args.n_bootstrap), encoding="utf-8")
    outputs["manifest_json"].write_text(
        json.dumps(
            {
                "n_bootstrap": int(args.n_bootstrap),
                "seed": int(args.seed),
                "fixed_local_baseline": FIXED_LOCAL_BASELINE,
                "eo_baselines": list(EO_COLUMNS),
                "slice_ids": [slice_id for slice_id, _ in SLICE_ORDER],
                "block_units": ["hour", "day"],
                "n_rows_long": int(long_df.shape[0]),
                "n_rows_wide": int(wide_df.shape[0]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

