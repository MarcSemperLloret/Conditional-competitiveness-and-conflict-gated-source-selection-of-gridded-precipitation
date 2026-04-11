#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_temporal_sensitivity_2022_v1 import build_year_frame
import build_conflict_policy_extension_v1 as ext


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper_11_density_thresholds" / "results" / "conflict_policy_multiyear_v1"
YEAR_PAIRS = [(2019, 2020), (2020, 2021), (2021, 2022), (2022, 2023)]
HIGHLIGHT_THRESHOLDS_MM = [0.1, 1.0, 5.0, 10.0]


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


def build_year_cache() -> dict[int, pd.DataFrame]:
    years = sorted({year for pair in YEAR_PAIRS for year in pair})
    return {year: build_year_frame(year) for year in years}


def evaluate_pair(train_year: int, test_year: int, frames: dict[int, pd.DataFrame]) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    train_df = frames[train_year]
    test_df = frames[test_year]
    policy = ext.train_conflict_policy(train_df)
    policy["train_year"] = train_year
    policy["test_year"] = test_year

    performance_df, bootstrap_df = ext.evaluate_models(test_df, policy)
    alerts_df = ext.evaluate_alerts(test_df, policy)

    perf = performance_df.set_index("model")
    bootstrap_idx = bootstrap_df.set_index(["comparison", "slice"])
    alerts_idx = alerts_df.set_index(["threshold_mm", "model"])

    summary_row: dict[str, object] = {
        "train_year": train_year,
        "test_year": test_year,
        "conflict_threshold_q75_mm": float(policy["conflict_threshold_q75_mm"]),
        "low_conflict_choice": str(policy["low_conflict_choice"]),
        "high_conflict_choice": str(policy["high_conflict_choice"]),
        "policy_local_usage_share": float(perf.loc[ext.POLICY_LABEL, "local_usage_share"]),
        "policy_overall_mae_mm": float(perf.loc[ext.POLICY_LABEL, "overall_mae_mm"]),
        "policy_positive_rain_mae_mm": float(perf.loc[ext.POLICY_LABEL, "positive_rain_mae_mm"]),
        "euradclim_overall_mae_mm": float(perf.loc[ext.BEST_FIXED_EO_LABEL, "overall_mae_mm"]),
        "euradclim_positive_rain_mae_mm": float(perf.loc[ext.BEST_FIXED_EO_LABEL, "positive_rain_mae_mm"]),
        "fixed_local_overall_mae_mm": float(perf.loc[ext.LOCAL_LABEL, "overall_mae_mm"]),
        "fixed_local_positive_rain_mae_mm": float(perf.loc[ext.LOCAL_LABEL, "positive_rain_mae_mm"]),
        "delta_policy_minus_euradclim_overall_mm": float(
            bootstrap_idx.loc[(f"{ext.POLICY_LABEL} minus {ext.BEST_FIXED_EO_LABEL}", "overall"), "delta_mae_eo_minus_local"]
        ),
        "delta_policy_minus_euradclim_overall_ci_low": float(
            bootstrap_idx.loc[(f"{ext.POLICY_LABEL} minus {ext.BEST_FIXED_EO_LABEL}", "overall"), "ci_low"]
        ),
        "delta_policy_minus_euradclim_overall_ci_high": float(
            bootstrap_idx.loc[(f"{ext.POLICY_LABEL} minus {ext.BEST_FIXED_EO_LABEL}", "overall"), "ci_high"]
        ),
        "delta_policy_minus_fixed_local_overall_mm": float(
            bootstrap_idx.loc[(f"{ext.POLICY_LABEL} minus {ext.LOCAL_LABEL}", "overall"), "delta_mae_eo_minus_local"]
        ),
        "delta_policy_minus_fixed_local_overall_ci_low": float(
            bootstrap_idx.loc[(f"{ext.POLICY_LABEL} minus {ext.LOCAL_LABEL}", "overall"), "ci_low"]
        ),
        "delta_policy_minus_fixed_local_overall_ci_high": float(
            bootstrap_idx.loc[(f"{ext.POLICY_LABEL} minus {ext.LOCAL_LABEL}", "overall"), "ci_high"]
        ),
        "delta_policy_minus_fixed_local_positive_rain_mm": float(
            bootstrap_idx.loc[(f"{ext.POLICY_LABEL} minus {ext.LOCAL_LABEL}", "positive_rain"), "delta_mae_eo_minus_local"]
        ),
        "delta_policy_minus_fixed_local_positive_rain_ci_low": float(
            bootstrap_idx.loc[(f"{ext.POLICY_LABEL} minus {ext.LOCAL_LABEL}", "positive_rain"), "ci_low"]
        ),
        "delta_policy_minus_fixed_local_positive_rain_ci_high": float(
            bootstrap_idx.loc[(f"{ext.POLICY_LABEL} minus {ext.LOCAL_LABEL}", "positive_rain"), "ci_high"]
        ),
    }

    for threshold in HIGHLIGHT_THRESHOLDS_MM:
        threshold_key = str(threshold).replace(".", "p")
        summary_row[f"policy_csi_{threshold_key}mm"] = float(alerts_idx.loc[(threshold, ext.POLICY_LABEL), "csi"])
        summary_row[f"euradclim_csi_{threshold_key}mm"] = float(alerts_idx.loc[(threshold, ext.BEST_FIXED_EO_LABEL), "csi"])
        summary_row[f"fixed_local_csi_{threshold_key}mm"] = float(alerts_idx.loc[(threshold, ext.LOCAL_LABEL), "csi"])

    bootstrap_df = bootstrap_df.copy()
    bootstrap_df.insert(0, "train_year", train_year)
    bootstrap_df.insert(1, "test_year", test_year)
    alerts_df = alerts_df.copy()
    alerts_df.insert(0, "train_year", train_year)
    alerts_df.insert(1, "test_year", test_year)
    return summary_row, bootstrap_df, alerts_df


def plot_multiyear(summary_df: pd.DataFrame, alerts_df: pd.DataFrame) -> None:
    test_years = summary_df["test_year"].to_numpy(dtype=int, copy=False)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))

    axes[0].plot(test_years, summary_df["policy_overall_mae_mm"], marker="o", linewidth=2.2, color=ext.COLORS[ext.POLICY_LABEL], label=ext.POLICY_LABEL)
    axes[0].plot(test_years, summary_df["euradclim_overall_mae_mm"], marker="o", linewidth=2.2, color=ext.COLORS[ext.BEST_FIXED_EO_LABEL], label=ext.BEST_FIXED_EO_LABEL)
    axes[0].plot(test_years, summary_df["fixed_local_overall_mae_mm"], marker="o", linewidth=2.2, color=ext.COLORS[ext.LOCAL_LABEL], label=ext.LOCAL_LABEL)
    axes[0].set_title("Overall MAE by held-out year")
    axes[0].set_xlabel("Held-out year")
    axes[0].set_ylabel("MAE (mm)")
    axes[0].set_xticks(test_years)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(test_years, summary_df["policy_positive_rain_mae_mm"], marker="o", linewidth=2.2, color=ext.COLORS[ext.POLICY_LABEL], label=ext.POLICY_LABEL)
    axes[1].plot(test_years, summary_df["euradclim_positive_rain_mae_mm"], marker="o", linewidth=2.2, color=ext.COLORS[ext.BEST_FIXED_EO_LABEL], label=ext.BEST_FIXED_EO_LABEL)
    axes[1].plot(test_years, summary_df["fixed_local_positive_rain_mae_mm"], marker="o", linewidth=2.2, color=ext.COLORS[ext.LOCAL_LABEL], label=ext.LOCAL_LABEL)
    axes[1].set_title("Positive-rain MAE by held-out year")
    axes[1].set_xlabel("Held-out year")
    axes[1].set_ylabel("MAE (mm)")
    axes[1].set_xticks(test_years)

    csi_delta = np.zeros((len(HIGHLIGHT_THRESHOLDS_MM), len(test_years)), dtype=float)
    for row_idx, threshold in enumerate(HIGHLIGHT_THRESHOLDS_MM):
        for col_idx, test_year in enumerate(test_years):
            subset = alerts_df[(alerts_df["test_year"] == test_year) & (alerts_df["threshold_mm"] == threshold)].set_index("model")
            csi_delta[row_idx, col_idx] = float(subset.loc[ext.POLICY_LABEL, "csi"] - subset.loc[ext.LOCAL_LABEL, "csi"])
    image = axes[2].imshow(csi_delta, aspect="auto", cmap="RdYlGn", vmin=-0.03, vmax=0.06)
    axes[2].set_title("Policy CSI gain over fixed local")
    axes[2].set_xlabel("Held-out year")
    axes[2].set_ylabel("Threshold (mm)")
    axes[2].set_xticks(np.arange(len(test_years)))
    axes[2].set_xticklabels(test_years)
    axes[2].set_yticks(np.arange(len(HIGHLIGHT_THRESHOLDS_MM)))
    axes[2].set_yticklabels(HIGHLIGHT_THRESHOLDS_MM)
    for row_idx in range(csi_delta.shape[0]):
        for col_idx in range(csi_delta.shape[1]):
            axes[2].text(col_idx, row_idx, f"{csi_delta[row_idx, col_idx]:+.03f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes[:2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_conflict_policy_multiyear_v1.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure_conflict_policy_multiyear_v1.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_multiyear_mae(summary_df: pd.DataFrame) -> None:
    test_years = summary_df["test_year"].to_numpy(dtype=int, copy=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))

    axes[0].plot(test_years, summary_df["policy_overall_mae_mm"], marker="o", linewidth=2.4, color=ext.COLORS[ext.POLICY_LABEL], label=ext.POLICY_LABEL)
    axes[0].plot(test_years, summary_df["euradclim_overall_mae_mm"], marker="o", linewidth=2.4, color=ext.COLORS[ext.BEST_FIXED_EO_LABEL], label=ext.BEST_FIXED_EO_LABEL)
    axes[0].plot(test_years, summary_df["fixed_local_overall_mae_mm"], marker="o", linewidth=2.4, color=ext.COLORS[ext.LOCAL_LABEL], label=ext.LOCAL_LABEL)
    axes[0].set_title("Overall MAE by held-out year")
    axes[0].set_xlabel("Held-out year")
    axes[0].set_ylabel("MAE (mm)")
    axes[0].set_xticks(test_years)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(test_years, summary_df["policy_positive_rain_mae_mm"], marker="o", linewidth=2.4, color=ext.COLORS[ext.POLICY_LABEL], label=ext.POLICY_LABEL)
    axes[1].plot(test_years, summary_df["euradclim_positive_rain_mae_mm"], marker="o", linewidth=2.4, color=ext.COLORS[ext.BEST_FIXED_EO_LABEL], label=ext.BEST_FIXED_EO_LABEL)
    axes[1].plot(test_years, summary_df["fixed_local_positive_rain_mae_mm"], marker="o", linewidth=2.4, color=ext.COLORS[ext.LOCAL_LABEL], label=ext.LOCAL_LABEL)
    axes[1].set_title("Positive-rain MAE by held-out year")
    axes[1].set_xlabel("Held-out year")
    axes[1].set_ylabel("MAE (mm)")
    axes[1].set_xticks(test_years)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_conflict_policy_multiyear_mae_v1.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure_conflict_policy_multiyear_mae_v1.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_multiyear_csi(alerts_df: pd.DataFrame) -> None:
    test_years = sorted(alerts_df["test_year"].unique())
    csi_delta = np.zeros((len(HIGHLIGHT_THRESHOLDS_MM), len(test_years)), dtype=float)
    for row_idx, threshold in enumerate(HIGHLIGHT_THRESHOLDS_MM):
        for col_idx, test_year in enumerate(test_years):
            subset = alerts_df[(alerts_df["test_year"] == test_year) & (alerts_df["threshold_mm"] == threshold)].set_index("model")
            csi_delta[row_idx, col_idx] = float(subset.loc[ext.POLICY_LABEL, "csi"] - subset.loc[ext.LOCAL_LABEL, "csi"])

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    image = ax.imshow(csi_delta, aspect="auto", cmap="RdYlGn", vmin=-0.03, vmax=0.06)
    ax.set_title("Policy CSI gain over fixed local")
    ax.set_xlabel("Held-out year")
    ax.set_ylabel("Threshold (mm)")
    ax.set_xticks(np.arange(len(test_years)))
    ax.set_xticklabels(test_years)
    ax.set_yticks(np.arange(len(HIGHLIGHT_THRESHOLDS_MM)))
    ax.set_yticklabels(HIGHLIGHT_THRESHOLDS_MM)
    for row_idx in range(csi_delta.shape[0]):
        for col_idx in range(csi_delta.shape[1]):
            ax.text(col_idx, row_idx, f"{csi_delta[row_idx, col_idx]:+.03f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_conflict_policy_multiyear_csi_v1.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure_conflict_policy_multiyear_csi_v1.pdf", bbox_inches="tight")
    plt.close(fig)


def build_notes(summary_df: pd.DataFrame, alerts_df: pd.DataFrame) -> str:
    lines = [
        "# Conflict Policy Multi-year Validation v1",
        "",
        "- Purpose: test whether the conflict-aware routing rule generalizes under rolling one-year-ahead validation.",
        f"- Year pairs: {', '.join(f'{train}->{test}' for train, test in YEAR_PAIRS)}.",
        "",
        "## Learned routing structure by split",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- {int(row['train_year'])}->{int(row['test_year'])}: q75 conflict threshold = {fmt(row['conflict_threshold_q75_mm'], 6)} mm; "
            f"low conflict -> {row['low_conflict_choice']}; high conflict -> {row['high_conflict_choice']}; "
            f"local usage on held-out year = {fmt(100.0 * row['policy_local_usage_share'], 1)}%."
        )

    lines.extend(["", "## Held-out performance summary"])
    for _, row in summary_df.iterrows():
        lines.append(
            f"- {int(row['test_year'])}: policy overall MAE = {fmt(row['policy_overall_mae_mm'])}, "
            f"EURADCLIM = {fmt(row['euradclim_overall_mae_mm'])}, fixed local = {fmt(row['fixed_local_overall_mae_mm'])}; "
            f"policy minus EURADCLIM = {fmt(row['delta_policy_minus_euradclim_overall_mm'])} "
            f"[{fmt(row['delta_policy_minus_euradclim_overall_ci_low'])}, {fmt(row['delta_policy_minus_euradclim_overall_ci_high'])}], "
            f"policy minus fixed local = {fmt(row['delta_policy_minus_fixed_local_overall_mm'])} "
            f"[{fmt(row['delta_policy_minus_fixed_local_overall_ci_low'])}, {fmt(row['delta_policy_minus_fixed_local_overall_ci_high'])}]."
        )

    lines.extend(["", "## Alert CSI gains over fixed local"])
    for threshold in HIGHLIGHT_THRESHOLDS_MM:
        threshold_rows = []
        for test_year in summary_df["test_year"]:
            subset = alerts_df[(alerts_df["test_year"] == test_year) & (alerts_df["threshold_mm"] == threshold)].set_index("model")
            delta = float(subset.loc[ext.POLICY_LABEL, "csi"] - subset.loc[ext.LOCAL_LABEL, "csi"])
            threshold_rows.append(f"{int(test_year)}: {delta:+.03f}")
        lines.append(f"- {fmt(threshold, 1)} mm: " + "; ".join(threshold_rows))
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = build_year_cache()

    summary_rows: list[dict[str, object]] = []
    bootstrap_frames: list[pd.DataFrame] = []
    alert_frames: list[pd.DataFrame] = []
    for train_year, test_year in YEAR_PAIRS:
        summary_row, bootstrap_df, alerts_df = evaluate_pair(train_year, test_year, frames)
        summary_rows.append(summary_row)
        bootstrap_frames.append(bootstrap_df)
        alert_frames.append(alerts_df)

    summary_df = pd.DataFrame(summary_rows).sort_values("test_year").reset_index(drop=True)
    bootstrap_df = pd.concat(bootstrap_frames, ignore_index=True)
    alerts_df = pd.concat(alert_frames, ignore_index=True)

    summary_df.to_csv(OUT_DIR / "conflict_policy_multiyear_summary_v1.csv", index=False)
    bootstrap_df.to_csv(OUT_DIR / "conflict_policy_multiyear_bootstrap_v1.csv", index=False)
    alerts_df.to_csv(OUT_DIR / "conflict_policy_multiyear_alerts_v1.csv", index=False)
    (OUT_DIR / "conflict_policy_multiyear_notes_v1.md").write_text(build_notes(summary_df, alerts_df), encoding="utf-8")
    plot_multiyear(summary_df, alerts_df)
    plot_multiyear_mae(summary_df)
    plot_multiyear_csi(alerts_df)
    (OUT_DIR / "conflict_policy_multiyear_manifest_v1.json").write_text(
        json.dumps(
            {
                "year_pairs": YEAR_PAIRS,
                "highlight_thresholds_mm": HIGHLIGHT_THRESHOLDS_MM,
                "bootstrap_resamples": ext.N_BOOTSTRAP,
                "seed": ext.SEED,
                "fixed_local_baseline": ext.base.FIXED_LOCAL_BASELINE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
