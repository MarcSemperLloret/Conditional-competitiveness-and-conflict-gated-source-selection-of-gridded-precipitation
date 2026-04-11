#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SOURCE_CSV = Path(r"d:\1_AVAMET_2.0_REVISIO\paper_10_fusion_fail\results\baselines_v3\baseline_results_v3.csv")
OUTPUT_DIR = Path(r"d:\1_AVAMET_2.0_REVISIO\paper_11_density_thresholds\results")

EO_BASELINES = ["source_imerg", "source_era5", "source_euradclim"]
LOCAL_BASELINES = [
    "domain_idw_leavecell_knn08",
    "domain_idw_leavecell_knn64",
    "domain_idw_leavecell_knn64_r15km",
]

KEEP_STRATIFIERS = [
    "overall",
    "support_regime",
    "observable_conflict",
    "target_intensity",
    "elevation",
    "coast",
]

STRATUM_ORDERS = {
    "overall": {"all": 0},
    "support_regime": {"single_station": 0, "robust_ge_2": 1},
    "observable_conflict": {"low_le_p75": 0, "mid_p75_p95": 1, "tail_gt_p95": 2},
    "target_intensity": {
        "=0": 0,
        "(0,0.1]": 1,
        "(0.1,1]": 2,
        "(1,5]": 3,
        "(5,10]": 4,
        "(10,20]": 5,
        ">20": 6,
    },
    "elevation": {"<200": 0, "200-600": 1, "600-1200": 2, ">=1200": 3},
    "coast": {"<10km": 0, "10-30km": 1, "30-60km": 2, ">=60km": 3},
}


def selection_label(scenario: str, stratifier: str) -> str:
    primary = scenario == "robust_ge_2" and stratifier in {
        "overall",
        "observable_conflict",
        "target_intensity",
        "elevation",
        "coast",
    }
    return "primary" if primary else "sensitivity"


def baseline_sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["mae_mm", "rmse_mm", "baseline"], na_position="last")


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["scenario", "scope", "stratifier", "stratum"]
    for keys, group in df.groupby(group_cols, dropna=False):
        eo_group = baseline_sort_frame(group[group["baseline"].isin(EO_BASELINES)].copy())
        local_group = baseline_sort_frame(group[group["baseline"].isin(LOCAL_BASELINES)].copy())
        if eo_group.empty or local_group.empty:
            continue

        best_eo = eo_group.iloc[0]
        best_local = local_group.iloc[0]
        scenario, scope, stratifier, stratum = keys
        local_mae = float(best_local["mae_mm"])
        eo_mae = float(best_eo["mae_mm"])
        delta = eo_mae - local_mae

        rows.append(
            {
                "selection": selection_label(str(scenario), str(stratifier)),
                "scenario": scenario,
                "scope": scope,
                "stratifier": stratifier,
                "stratum": stratum,
                "n_requested": int(best_local["n_requested"]),
                "best_eo": best_eo["baseline"],
                "best_eo_mae_mm": eo_mae,
                "best_eo_rmse_mm": float(best_eo["rmse_mm"]),
                "best_eo_corr": None if pd.isna(best_eo["corr"]) else float(best_eo["corr"]),
                "best_eo_coverage_share": float(best_eo["coverage_share"]),
                "best_local": best_local["baseline"],
                "best_local_mae_mm": local_mae,
                "best_local_rmse_mm": float(best_local["rmse_mm"]),
                "best_local_corr": None if pd.isna(best_local["corr"]) else float(best_local["corr"]),
                "best_local_coverage_share": float(best_local["coverage_share"]),
                "delta_mae_mm": delta,
                "delta_mae_pct_vs_local": None if local_mae == 0 else 100.0 * delta / local_mae,
                "eo_wins_on_mae": bool(eo_mae < local_mae),
            }
        )

    out = pd.DataFrame(rows)
    out["scenario_order"] = out["scenario"].map({"robust_ge_2": 0, "wide_ge_1": 1}).fillna(99)
    out["stratifier_order"] = out["stratifier"].map(
        {
            "overall": 0,
            "support_regime": 1,
            "observable_conflict": 2,
            "target_intensity": 3,
            "elevation": 4,
            "coast": 5,
        }
    ).fillna(99)
    out["stratum_order"] = out.apply(
        lambda row: STRATUM_ORDERS.get(str(row["stratifier"]), {}).get(str(row["stratum"]), 99),
        axis=1,
    )
    out = out.sort_values(["selection", "scenario_order", "stratifier_order", "stratum_order"]).reset_index(drop=True)
    return out.drop(columns=["scenario_order", "stratifier_order", "stratum_order"])


def build_markdown(table_df: pd.DataFrame) -> str:
    lines = [
        "# Best EO vs Best Local by Regime v1",
        "",
        f"- Source CSV: `{SOURCE_CSV}`",
        f"- EO baselines: `{EO_BASELINES}`",
        f"- Local baselines: `{LOCAL_BASELINES}`",
        "",
        "## Primary rows",
    ]
    primary = table_df[table_df["selection"] == "primary"].copy()
    if primary.empty:
        lines.append("- No primary rows were generated.")
    else:
        for _, row in primary.iterrows():
            lines.append(
                f"- `{row['scenario']}` / `{row['stratifier']}` / `{row['stratum']}`: "
                f"best EO = `{row['best_eo']}` ({row['best_eo_mae_mm']:.4f}), "
                f"best local = `{row['best_local']}` ({row['best_local_mae_mm']:.4f}), "
                f"delta MAE = {row['delta_mae_mm']:.4f}"
            )

    lines.extend(["", "## Winner counts by stratifier"])
    counts = (
        primary.groupby(["stratifier", "best_eo"])
        .size()
        .reset_index(name="n_rows")
        .sort_values(["stratifier", "n_rows", "best_eo"], ascending=[True, False, True])
    )
    if counts.empty:
        lines.append("- No winner counts available.")
    else:
        for _, row in counts.iterrows():
            lines.append(f"- `{row['stratifier']}`: `{row['best_eo']}` wins {int(row['n_rows'])} rows")

    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE_CSV)
    df = df[df["stratifier"].isin(KEEP_STRATIFIERS)].copy()
    df = df[df["baseline"].isin(EO_BASELINES + LOCAL_BASELINES)].copy()

    table_df = build_table(df)
    markdown = build_markdown(table_df)

    csv_path = OUTPUT_DIR / "best_eo_vs_best_local_by_regime_v1.csv"
    md_path = OUTPUT_DIR / "best_eo_vs_best_local_by_regime_v1.md"
    meta_path = OUTPUT_DIR / "best_eo_vs_best_local_by_regime_v1.json"

    table_df.to_csv(csv_path, index=False)
    md_path.write_text(markdown, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "source_csv": str(SOURCE_CSV),
                "eo_baselines": EO_BASELINES,
                "local_baselines": LOCAL_BASELINES,
                "keep_stratifiers": KEEP_STRATIFIERS,
                "n_rows": int(table_df.shape[0]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
