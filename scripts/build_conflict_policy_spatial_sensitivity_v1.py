#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

import build_conflict_policy_extension_v1 as ext
import build_delta_mae_by_support_score_and_regime_v1 as base
from build_temporal_sensitivity_2022_v1 import build_year_frame


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "conflict_policy_spatial_sensitivity_v1"
FIG_DPI = 400

BG = "#ffffff"
GRID = "#d9e1ea"
INK = "#1f2a33"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Aptos", "Segoe UI", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
        "savefig.dpi": FIG_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


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


def save_exports(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.png")
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    plt.close(fig)


def build_csi_block_sensitivity(test_df: pd.DataFrame, policy: dict[str, object]) -> pd.DataFrame:
    hour_df = ext.evaluate_alert_bootstrap(test_df, policy, block_unit="hour").copy()
    day_df = ext.evaluate_alert_bootstrap(test_df, policy, block_unit="day").copy()
    merge_keys = ["comparison", "threshold_mm"]
    wide = hour_df.merge(day_df, on=merge_keys, suffixes=("_hour", "_day"))
    wide["sign_match"] = np.sign(wide["delta_csi_left_minus_right_hour"]) == np.sign(wide["delta_csi_left_minus_right_day"])
    wide["ci_exclusion_match"] = wide["ci_excludes_zero_hour"] == wide["ci_excludes_zero_day"]
    return wide.sort_values(["comparison", "threshold_mm"]).reset_index(drop=True)


def build_spatial_summary(test_df: pd.DataFrame, policy: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    robust = test_df["avamet_n_active"].to_numpy(dtype=np.int16, copy=False) >= 2
    y_true = test_df["avamet_agg_mm"].to_numpy(dtype=np.float64, copy=False)[robust]
    local_pred = test_df.loc[robust, base.FIXED_LOCAL_BASELINE].to_numpy(dtype=np.float64, copy=False)
    eurad_pred = test_df.loc[robust, "euradclim_on_imerg_mm"].to_numpy(dtype=np.float64, copy=False)
    policy_pred, policy_local_used = ext.build_policy_predictions(test_df, policy)

    cell_df = pd.DataFrame(
        {
            "grid_id": test_df.loc[robust, "grid_id"].to_numpy(),
            "target_mm": y_true,
            "observable_conflict_mm": test_df.loc[robust, "observable_conflict_nonref_v0"].to_numpy(dtype=np.float64, copy=False),
            "policy_local_used": policy_local_used[robust].astype(np.float64),
            "eurad_delta_mae_mean": np.abs(eurad_pred - y_true) - np.abs(local_pred - y_true),
            "policy_delta_mae_mean": np.abs(policy_pred[robust] - y_true) - np.abs(local_pred - y_true),
        }
    )
    cell_df = (
        cell_df.groupby("grid_id", as_index=False)
        .agg(
            n_rows=("grid_id", "size"),
            mean_target_mm=("target_mm", "mean"),
            mean_conflict_mm=("observable_conflict_mm", "mean"),
            policy_local_usage_share=("policy_local_used", "mean"),
            eurad_delta_mae_mean=("eurad_delta_mae_mean", "mean"),
            policy_delta_mae_mean=("policy_delta_mae_mean", "mean"),
        )
        .sort_values("grid_id")
        .reset_index(drop=True)
    )

    static_df = pd.read_parquet(base.STATIC_PARQUET)
    static_df = static_df.loc[static_df["inside_cv_polygon"].fillna(False).astype(bool)].copy()
    static_df = static_df[["grid_id", "lon", "lat", "elevation_m", "distance_to_coast_km"]]
    cell_df = cell_df.merge(static_df, on="grid_id", how="left")

    coast_codes, coast_edges = pd.qcut(cell_df["distance_to_coast_km"], q=3, labels=False, retbins=True, duplicates="drop")
    coast_codes = coast_codes.astype(int)
    coast_labels = []
    for idx in range(len(coast_edges) - 1):
        left = coast_edges[idx]
        right = coast_edges[idx + 1]
        if idx == 0:
            coast_labels.append(f"Coastal [{left:.1f}, {right:.1f}] km")
        elif idx == len(coast_edges) - 2:
            coast_labels.append(f"Inland ({left:.1f}, {right:.1f}] km")
        else:
            coast_labels.append(f"Transition ({left:.1f}, {right:.1f}] km")
    cell_df["coast_bin"] = pd.Categorical.from_codes(coast_codes, categories=coast_labels, ordered=True)

    coast_rows: list[dict[str, object]] = []
    for coast_bin, group in cell_df.groupby("coast_bin", observed=False):
        coast_rows.append(
            {
                "coast_bin": str(coast_bin),
                "n_cells": int(group.shape[0]),
                "n_rows": int(group["n_rows"].sum()),
                "median_eurad_delta_mae_mean": float(group["eurad_delta_mae_mean"].median()),
                "eurad_better_cell_share": float((group["eurad_delta_mae_mean"] < 0.0).mean()),
                "median_policy_delta_mae_mean": float(group["policy_delta_mae_mean"].median()),
                "policy_better_cell_share": float((group["policy_delta_mae_mean"] < 0.0).mean()),
            }
        )
    coast_summary_df = pd.DataFrame(coast_rows)
    overall = pd.DataFrame(
        [
            {
                "coast_bin": "Overall",
                "n_cells": int(cell_df.shape[0]),
                "n_rows": int(cell_df["n_rows"].sum()),
                "median_eurad_delta_mae_mean": float(cell_df["eurad_delta_mae_mean"].median()),
                "eurad_better_cell_share": float((cell_df["eurad_delta_mae_mean"] < 0.0).mean()),
                "median_policy_delta_mae_mean": float(cell_df["policy_delta_mae_mean"].median()),
                "policy_better_cell_share": float((cell_df["policy_delta_mae_mean"] < 0.0).mean()),
            }
        ]
    )
    coast_summary_df = pd.concat([overall, coast_summary_df], ignore_index=True)
    coast_summary_df["n_cells"] = coast_summary_df["n_cells"].astype(int)
    coast_summary_df["n_rows"] = coast_summary_df["n_rows"].astype(int)
    return cell_df, coast_summary_df


def plot_spatial_delta_maps(cell_df: pd.DataFrame) -> None:
    max_abs = float(
        max(
            abs(cell_df["eurad_delta_mae_mean"].min()),
            abs(cell_df["eurad_delta_mae_mean"].max()),
            abs(cell_df["policy_delta_mae_mean"].min()),
            abs(cell_df["policy_delta_mae_mean"].max()),
        )
    )
    max_abs = max(max_abs, 0.005)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.9))
    fig.subplots_adjust(top=0.86, wspace=0.28)
    panels = [
        ("eurad_delta_mae_mean", "A. EURADCLIM minus fixed local"),
        ("policy_delta_mae_mean", "B. Policy minus fixed local"),
    ]
    for ax, (value_col, title) in zip(axes, panels, strict=True):
        sc = ax.scatter(
            cell_df["lon"],
            cell_df["lat"],
            c=cell_df[value_col],
            cmap="RdBu_r",
            norm=norm,
            s=205,
            marker="s",
            linewidths=0.0,
            alpha=0.96,
        )
        ax.set_title(title, loc="left", pad=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color=GRID, linewidth=0.6, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#b7c1cb")
        ax.spines["bottom"].set_color("#b7c1cb")
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Cell-level mean Î”MAE (mm)")

    fig.suptitle("Spatial heterogeneity of source-selection gains in 2023", x=0.07, y=0.97, ha="left", fontsize=15, fontweight="bold")
    save_exports(fig, "figure_conflict_policy_spatial_delta_v1")


def build_notes(csi_df: pd.DataFrame, cell_df: pd.DataFrame, coast_summary_df: pd.DataFrame) -> str:
    lines = [
        "# Conflict Policy Spatial Sensitivity v1",
        "",
        "- Purpose: add a stricter day-block CSI sensitivity and a lightweight spatial disaggregation of the 2023 source-selection result.",
        "",
        "## CSI block-length sensitivity",
    ]
    for _, row in csi_df.iterrows():
        lines.append(
            f"- {row['comparison']} / {fmt(row['threshold_mm'], 1)} mm: "
            f"hour-block Î”CSI = {fmt(row['delta_csi_left_minus_right_hour'])} "
            f"[{fmt(row['ci_low_hour'])}, {fmt(row['ci_high_hour'])}], "
            f"day-block Î”CSI = {fmt(row['delta_csi_left_minus_right_day'])} "
            f"[{fmt(row['ci_low_day'])}, {fmt(row['ci_high_day'])}]."
        )
    lines.extend(["", "## Spatial cell summary"])
    lines.append(
        f"- Robust-support cells with at least one evaluated row: {int(cell_df.shape[0])}."
    )
    lines.append(
        f"- EURADCLIM beats fixed local in {int((cell_df['eurad_delta_mae_mean'] < 0.0).sum())} cells; "
        f"policy beats fixed local in {int((cell_df['policy_delta_mae_mean'] < 0.0).sum())} cells."
    )
    for _, row in coast_summary_df.iterrows():
        lines.append(
            f"- {row['coast_bin']}: median EURADCLIM Î”MAE = {fmt(row['median_eurad_delta_mae_mean'])}, "
            f"EURADCLIM better-cell share = {fmt(100.0 * row['eurad_better_cell_share'], 1)}%; "
            f"median policy Î”MAE = {fmt(row['median_policy_delta_mae_mean'])}, "
            f"policy better-cell share = {fmt(100.0 * row['policy_better_cell_share'], 1)}%."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = build_year_frame(2022)
    test_df = build_year_frame(2023)
    policy = ext.train_conflict_policy(train_df)

    csi_df = build_csi_block_sensitivity(test_df, policy)
    cell_df, coast_summary_df = build_spatial_summary(test_df, policy)
    plot_spatial_delta_maps(cell_df)

    csi_path = OUT_DIR / "conflict_policy_alert_block_sensitivity_v1.csv"
    cell_path = OUT_DIR / "conflict_policy_spatial_cell_summary_v1.csv"
    coast_path = OUT_DIR / "conflict_policy_spatial_coast_summary_v1.csv"
    notes_path = OUT_DIR / "conflict_policy_spatial_notes_v1.md"
    manifest_path = OUT_DIR / "conflict_policy_spatial_manifest_v1.json"

    csi_df.to_csv(csi_path, index=False)
    cell_df.to_csv(cell_path, index=False)
    coast_summary_df.to_csv(coast_path, index=False)
    notes_path.write_text(build_notes(csi_df, cell_df, coast_summary_df), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "outputs": {
                    "conflict_policy_alert_block_sensitivity_v1.csv": str(csi_path),
                    "conflict_policy_spatial_cell_summary_v1.csv": str(cell_path),
                    "conflict_policy_spatial_coast_summary_v1.csv": str(coast_path),
                    "figure_conflict_policy_spatial_delta_v1.png": str(OUT_DIR / "figure_conflict_policy_spatial_delta_v1.png"),
                    "figure_conflict_policy_spatial_delta_v1.pdf": str(OUT_DIR / "figure_conflict_policy_spatial_delta_v1.pdf"),
                    "conflict_policy_spatial_notes_v1.md": str(notes_path),
                },
                "train_year": 2022,
                "test_year": 2023,
                "seed": ext.SEED,
                "bootstrap_resamples": ext.N_BOOTSTRAP,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

