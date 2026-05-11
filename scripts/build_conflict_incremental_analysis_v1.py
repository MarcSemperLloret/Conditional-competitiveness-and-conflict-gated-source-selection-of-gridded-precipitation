#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import build_delta_mae_by_support_score_and_regime_v1 as base
from local_support_metrics_v1 import build_year_prediction_frame_with_support


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "conflict_incremental_analysis_v1"
TRAIN_YEAR = 2022
TEST_YEAR = 2023

EO_SPECS = [
    ("IMERG", "imerg_mm"),
    ("ERA5", "era5_mm"),
    ("EURADCLIM", "euradclim_on_imerg_mm"),
]
MODEL_ORDER = [
    "target_intensity_only",
    "target_intensity_plus_support",
    "target_intensity_plus_support_plus_conflict",
]
MODEL_LABELS = {
    "target_intensity_only": "Target intensity",
    "target_intensity_plus_support": "Target intensity + support",
    "target_intensity_plus_support_plus_conflict": "Target intensity + support + conflict",
}
MODEL_COLORS = {
    "target_intensity_only": "#aab5c0",
    "target_intensity_plus_support": "#4f7cac",
    "target_intensity_plus_support_plus_conflict": "#2f8d62",
}
CONFLICT_BIN_ORDER = {"low": 0, "mid": 1, "tail": 2}
CONFLICT_BIN_LABELS = {"low": "Low", "mid": "Mid", "tail": "Tail"}
PRODUCT_COLORS = {"IMERG": "#1677b3", "ERA5": "#e18c31", "EURADCLIM": "#2f8d62"}
FIG_DPI = 400

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Aptos", "Segoe UI", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
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


def robust_subset(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["avamet_n_active"] >= 2].copy().reset_index(drop=True)


def error_gap(df: pd.DataFrame, eo_col: str) -> np.ndarray:
    eo_err = np.abs(df[eo_col].to_numpy(dtype=np.float64, copy=False) - df["avamet_agg_mm"].to_numpy(dtype=np.float64, copy=False))
    local_err = np.abs(
        df[base.FIXED_LOCAL_BASELINE].to_numpy(dtype=np.float64, copy=False)
        - df["avamet_agg_mm"].to_numpy(dtype=np.float64, copy=False)
    )
    return eo_err - local_err


def build_feature_blocks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    target_intensity = np.log1p(np.clip(df["avamet_agg_mm"].to_numpy(dtype=np.float64, copy=False), 0.0, None))
    conflict = np.log1p(
        np.clip(df["observable_conflict_nonref_v0"].to_numpy(dtype=np.float64, copy=False), 0.0, None)
    )
    support = np.column_stack(
        [
            np.log1p(df["avamet_n_active"].to_numpy(dtype=np.float64, copy=False)),
            np.log1p(df["leavecell_active_15km"].to_numpy(dtype=np.float64, copy=False)),
            np.log1p(np.clip(df["leavecell_idw_neff"].to_numpy(dtype=np.float64, copy=False), 0.0, None)),
        ]
    )
    return {
        "target_intensity_only": np.column_stack([target_intensity, np.square(target_intensity)]),
        "target_intensity_plus_support": np.column_stack([target_intensity, np.square(target_intensity), support]),
        "target_intensity_plus_support_plus_conflict": np.column_stack(
            [target_intensity, np.square(target_intensity), support, conflict, np.square(conflict)]
        ),
    }


def fit_standardized_ols(features: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    mu = features.mean(axis=0)
    sd = features.std(axis=0)
    sd[sd == 0.0] = 1.0
    design = np.column_stack([np.ones(features.shape[0]), (features - mu) / sd])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return {"mu": mu, "sd": sd, "beta": beta}


def predict_standardized_ols(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(features.shape[0]), (features - model["mu"]) / model["sd"]])
    return design @ model["beta"]


def corr_or_nan(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def build_nested_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    train_features = build_feature_blocks(train_df)
    test_features = build_feature_blocks(test_df)
    q75 = float(train_df["observable_conflict_nonref_v0"].quantile(0.75))
    q95 = float(train_df["observable_conflict_nonref_v0"].quantile(0.95))

    rows: list[dict[str, object]] = []
    residual_rows: list[pd.DataFrame] = []
    residual_pred_store: dict[str, np.ndarray] = {}

    for eo_label, eo_col in EO_SPECS:
        y_train = error_gap(train_df, eo_col)
        y_test = error_gap(test_df, eo_col)
        prev_r2: float | None = None

        for model_name in MODEL_ORDER:
            model = fit_standardized_ols(train_features[model_name], y_train)
            pred = predict_standardized_ols(model, test_features[model_name])
            ss_res = float(np.sum(np.square(y_test - pred)))
            ss_tot = float(np.sum(np.square(y_test - y_test.mean())))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
            rmse = float(np.sqrt(np.mean(np.square(y_test - pred))))
            rows.append(
                {
                    "eo_label": eo_label,
                    "model_name": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "train_year": TRAIN_YEAR,
                    "test_year": TEST_YEAR,
                    "n_train": int(y_train.size),
                    "n_test": int(y_test.size),
                    "heldout_r2": r2,
                    "heldout_rmse_gap_mm": rmse,
                    "heldout_corr": corr_or_nan(y_test, pred),
                    "delta_r2_vs_previous": None if prev_r2 is None else float(r2 - prev_r2),
                }
            )
            prev_r2 = r2
            if model_name == "target_intensity_plus_support":
                residual_pred_store[eo_label] = pred

        test_conflict = test_df["observable_conflict_nonref_v0"].to_numpy(dtype=np.float64, copy=False)
        conflict_bin = np.select(
            [
                test_conflict <= q75,
                (test_conflict > q75) & (test_conflict <= q95),
                test_conflict > q95,
            ],
            ["low", "mid", "tail"],
            default="unknown",
        )
        pred_support_only = residual_pred_store[eo_label]
        residual = y_test - pred_support_only
        residual_df = pd.DataFrame(
            {
                "eo_label": eo_label,
                "train_conflict_bin": conflict_bin,
                "observed_gap_mm": y_test,
                "residual_gap_mm": residual,
            }
        )
        summary = (
            residual_df.groupby("train_conflict_bin", observed=False)
            .agg(
                n_rows=("observed_gap_mm", "size"),
                mean_observed_gap_mm=("observed_gap_mm", "mean"),
                median_observed_gap_mm=("observed_gap_mm", "median"),
                mean_residual_gap_mm=("residual_gap_mm", "mean"),
                median_residual_gap_mm=("residual_gap_mm", "median"),
            )
            .reset_index()
        )
        summary["eo_label"] = eo_label
        residual_rows.append(summary)

    summary_df = pd.DataFrame(rows).sort_values(["eo_label", "model_name"]).reset_index(drop=True)
    residual_df = (
        pd.concat(residual_rows, ignore_index=True)
        .assign(conflict_order=lambda frame: frame["train_conflict_bin"].map(CONFLICT_BIN_ORDER))
        .sort_values(["eo_label", "conflict_order"])
        .drop(columns=["conflict_order"])
        .reset_index(drop=True)
    )
    meta = {"train_conflict_q75": q75, "train_conflict_q95": q95}
    return summary_df, residual_df, meta


def plot_incremental(summary_df: pd.DataFrame, residual_df: pd.DataFrame) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.4, 4.6))
    product_x = np.arange(len(EO_SPECS), dtype=float)
    bar_width = 0.22
    offsets = {
        "target_intensity_only": -bar_width,
        "target_intensity_plus_support": 0.0,
        "target_intensity_plus_support_plus_conflict": bar_width,
    }

    for model_name in MODEL_ORDER:
        sub = summary_df[summary_df["model_name"] == model_name].copy()
        heights = [
            float(sub.loc[sub["eo_label"] == eo_label, "heldout_r2"].iloc[0])
            for eo_label, _ in EO_SPECS
        ]
        ax_left.bar(
            product_x + offsets[model_name],
            heights,
            width=bar_width * 0.95,
            color=MODEL_COLORS[model_name],
            edgecolor="white",
            linewidth=0.8,
            label=MODEL_LABELS[model_name],
        )

    ax_left.set_xticks(product_x)
    ax_left.set_xticklabels([eo_label for eo_label, _ in EO_SPECS])
    ax_left.set_ylabel("Held-out $R^2$ on 2023 error-gap rows")
    ax_left.set_title("Conflict adds held-out explanatory value", loc="left", pad=10)
    ax_left.grid(axis="y", color="#d9e1ea", linewidth=0.8, alpha=0.9)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.set_axisbelow(True)

    conflict_x = np.arange(3, dtype=float)
    ax_right.axhline(0.0, color="#aab5c0", linewidth=1.1, linestyle="--")
    for eo_label, _ in EO_SPECS:
        sub = residual_df[residual_df["eo_label"] == eo_label].copy()
        y = [
            float(sub.loc[sub["train_conflict_bin"] == conflict_bin, "mean_residual_gap_mm"].iloc[0])
            for conflict_bin in ["low", "mid", "tail"]
        ]
        ax_right.plot(
            conflict_x,
            y,
            color=PRODUCT_COLORS[eo_label],
            linewidth=1.8,
            marker="o",
            markersize=5.5,
            markerfacecolor=PRODUCT_COLORS[eo_label],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=eo_label,
        )
    ax_right.set_xticks(conflict_x)
    ax_right.set_xticklabels([CONFLICT_BIN_LABELS[key] for key in ["low", "mid", "tail"]])
    ax_right.set_ylabel("Mean residual error-gap after intensity + support (mm)")
    ax_right.set_title("Residual local advantage still tracks conflict", loc="left", pad=10)
    ax_right.grid(axis="y", color="#d9e1ea", linewidth=0.8, alpha=0.9)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.set_axisbelow(True)

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    fig.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        frameon=False,
        ncol=3,
    )
    fig.suptitle("Incremental value of conflict beyond intensity and support", x=0.06, ha="left", y=1.10)
    save_exports(fig, "figure_conflict_incremental_analysis_v1")


def build_markdown(summary_df: pd.DataFrame, residual_df: pd.DataFrame, meta: dict[str, object]) -> str:
    lines = [
        "# Conflict Incremental Analysis v1",
        "",
        f"- Train year: `{TRAIN_YEAR}`",
        f"- Test year: `{TEST_YEAR}`",
        "- Purpose: test whether conflict retains held-out explanatory value for gridded-minus-local error gaps after controlling for realized target intensity and local-support strength.",
        (
            "- Support controls: `avamet_n_active`, active leave-cell gauges within `15 km`, "
            "and leave-cell IDW effective support."
        ),
        (
            "- Train-defined conflict thresholds applied to 2023 residual summaries: "
            f"q75 = `{fmt(meta['train_conflict_q75'])}`, q95 = `{fmt(meta['train_conflict_q95'])}`."
        ),
        "",
        "## Held-out nested-model summary",
    ]

    for eo_label, _ in EO_SPECS:
        sub = summary_df[summary_df["eo_label"] == eo_label].copy()
        lines.append(f"- `{eo_label}`")
        for _, row in sub.iterrows():
            gain = "baseline" if pd.isna(row["delta_r2_vs_previous"]) else f"delta R2 = {fmt(row['delta_r2_vs_previous'])}"
            lines.append(
                f"  - {row['model_label']}: held-out R2 = {fmt(row['heldout_r2'])}, "
                f"corr = {fmt(row['heldout_corr'])}, RMSE = {fmt(row['heldout_rmse_gap_mm'])}, {gain}"
            )

    lines.extend(["", "## Residual gap after removing intensity + support"])
    for eo_label, _ in EO_SPECS:
        sub = residual_df[residual_df["eo_label"] == eo_label].copy()
        summary = "; ".join(
            (
                f"{CONFLICT_BIN_LABELS[row['train_conflict_bin']]}: "
                f"obs = {fmt(row['mean_observed_gap_mm'])}, resid = {fmt(row['mean_residual_gap_mm'])}"
            )
            for _, row in sub.iterrows()
        )
        lines.append(f"- `{eo_label}`: {summary}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = robust_subset(build_year_prediction_frame_with_support(TRAIN_YEAR))
    test_df = robust_subset(build_year_prediction_frame_with_support(TEST_YEAR))
    summary_df, residual_df, meta = build_nested_summary(train_df, test_df)

    summary_path = OUT_DIR / "conflict_incremental_model_summary_v1.csv"
    residual_path = OUT_DIR / "conflict_incremental_residual_summary_v1.csv"
    md_path = OUT_DIR / "conflict_incremental_notes_v1.md"
    manifest_path = OUT_DIR / "conflict_incremental_manifest_v1.json"

    summary_df.to_csv(summary_path, index=False)
    residual_df.to_csv(residual_path, index=False)
    plot_incremental(summary_df, residual_df)
    md_path.write_text(build_markdown(summary_df, residual_df, meta), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "train_year": TRAIN_YEAR,
                "test_year": TEST_YEAR,
                "fixed_local_baseline": base.FIXED_LOCAL_BASELINE,
                "train_conflict_q75": meta["train_conflict_q75"],
                "train_conflict_q95": meta["train_conflict_q95"],
                "model_summary_csv": str(summary_path),
                "residual_summary_csv": str(residual_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

