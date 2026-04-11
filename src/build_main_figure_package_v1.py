#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_CSV = ROOT / "paper_11_density_thresholds" / "results" / "bootstrap_delta_mae_v1" / "bootstrap_summary.csv"
BEST_REGIME_CSV = ROOT / "paper_11_density_thresholds" / "results" / "best_eo_vs_best_local_by_regime_v1.csv"
OUTPUT_DIR = ROOT / "paper_11_density_thresholds" / "results" / "main_figure_v1"

EO_LABELS = {
    "source_imerg": "IMERG",
    "source_era5": "ERA5",
    "source_euradclim": "EURADCLIM",
}
SLICE_LABELS = {
    "overall_all": "Global",
    "overall_positive_target": "Positive rain",
    "conflict_low": "Low conflict",
    "conflict_mid": "Mid conflict",
    "conflict_tail": "High conflict",
}
STRATUM_LABELS = {
    "all": "Global",
    "low_le_p75": "Low conflict",
    "mid_p75_p95": "Mid conflict",
    "tail_gt_p95": "High conflict",
    "(0.1,1]": "0.1-1 mm",
    "(1,5]": "1-5 mm",
    "(5,10]": "5-10 mm",
    "(10,20]": "10-20 mm",
    ">20": ">20 mm",
}


def build_panel_a(bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    keep = bootstrap_df[
        (bootstrap_df["comparator_kind"] == "fixed_local")
        & (bootstrap_df["slice_id"].isin(["overall_all", "overall_positive_target"]))
    ].copy()
    keep["panel"] = "A"
    keep["eo_label"] = keep["eo_baseline"].map(EO_LABELS)
    keep["slice_label"] = keep["slice_id"].map(SLICE_LABELS)
    keep["slice_order"] = keep["slice_id"].map({"overall_all": 0, "overall_positive_target": 1})
    return keep.sort_values(["slice_order", "eo_label"]).reset_index(drop=True)


def build_panel_b(bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    keep = bootstrap_df[
        (bootstrap_df["comparator_kind"] == "fixed_local")
        & (bootstrap_df["slice_id"].isin(["conflict_low", "conflict_mid", "conflict_tail"]))
    ].copy()
    keep["panel"] = "B"
    keep["eo_label"] = keep["eo_baseline"].map(EO_LABELS)
    keep["slice_label"] = keep["slice_id"].map(SLICE_LABELS)
    keep["slice_order"] = keep["slice_id"].map({"conflict_low": 0, "conflict_mid": 1, "conflict_tail": 2})
    return keep.sort_values(["slice_order", "eo_label"]).reset_index(drop=True)


def build_panel_c(best_df: pd.DataFrame) -> pd.DataFrame:
    keep = best_df[
        (best_df["selection"] == "primary")
        & (best_df["scenario"] == "robust_ge_2")
        & (
            ((best_df["stratifier"] == "overall") & (best_df["stratum"] == "all"))
            | ((best_df["stratifier"] == "observable_conflict") & (best_df["stratum"].isin(["low_le_p75", "mid_p75_p95", "tail_gt_p95"])))
            | ((best_df["stratifier"] == "target_intensity") & (best_df["stratum"].isin(["(0.1,1]", "(1,5]", "(5,10]", "(10,20]", ">20"])))
        )
    ].copy()
    keep["panel"] = "C"
    keep["row_label"] = keep["stratum"].map(STRATUM_LABELS)
    keep["best_eo_label"] = keep["best_eo"].map(EO_LABELS)
    keep["row_group"] = keep["stratifier"].map(
        {
            "overall": "Overall",
            "observable_conflict": "Conflict",
            "target_intensity": "Intensity",
        }
    )
    order_map = {
        ("overall", "all"): 0,
        ("observable_conflict", "low_le_p75"): 1,
        ("observable_conflict", "mid_p75_p95"): 2,
        ("observable_conflict", "tail_gt_p95"): 3,
        ("target_intensity", "(0.1,1]"): 4,
        ("target_intensity", "(1,5]"): 5,
        ("target_intensity", "(5,10]"): 6,
        ("target_intensity", "(10,20]"): 7,
        ("target_intensity", ">20"): 8,
    }
    keep["row_order"] = keep.apply(lambda row: order_map[(row["stratifier"], row["stratum"])], axis=1)
    return keep.sort_values("row_order").reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bootstrap_df = pd.read_csv(BOOTSTRAP_CSV)
    best_df = pd.read_csv(BEST_REGIME_CSV)

    panel_a = build_panel_a(bootstrap_df)
    panel_b = build_panel_b(bootstrap_df)
    panel_c = build_panel_c(best_df)

    panel_a.to_csv(OUTPUT_DIR / "panel_a_global_and_positive_rain.csv", index=False)
    panel_b.to_csv(OUTPUT_DIR / "panel_b_conflict_bootstrap.csv", index=False)
    panel_c.to_csv(OUTPUT_DIR / "panel_c_best_eo_by_regime.csv", index=False)
    (OUTPUT_DIR / "figure_manifest.json").write_text(
        json.dumps(
            {
                "bootstrap_csv": str(BOOTSTRAP_CSV),
                "best_regime_csv": str(BEST_REGIME_CSV),
                "panels": {
                    "A": "Global and positive-rain delta MAE vs fixed local with 95% bootstrap intervals",
                    "B": "Conflict-conditioned delta MAE vs fixed local with 95% bootstrap intervals",
                    "C": "Best EO by regime and delta MAE vs best local",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
