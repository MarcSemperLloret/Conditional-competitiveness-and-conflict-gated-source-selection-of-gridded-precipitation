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
from build_temporal_sensitivity_2022_v1 import build_year_frame
from run_delta_mae_block_bootstrap_v1 import bootstrap_mae_difference, factorize_blocks


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper_11_density_thresholds" / "results" / "conflict_policy_extension_v1"
SEED = 42
N_BOOTSTRAP = 1500
ALERT_THRESHOLDS_MM = [0.1, 1.0, 5.0, 10.0]
THRESHOLD_QUANTILES = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
LOCAL_LABEL = "Fixed local"
BEST_FIXED_EO_LABEL = "EURADCLIM"
POLICY_LABEL = "Conflict policy"
MEDIAN_GATE_LABEL = "Median-conflict IMERG/local"
ALT_Q75_LABEL = "q75 EURADCLIM/local"

COLORS = {
    POLICY_LABEL: "#1f6f8b",
    BEST_FIXED_EO_LABEL: "#2f8d62",
    "ERA5": "#e18c31",
    "IMERG": "#1677b3",
    LOCAL_LABEL: "#374151",
}


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


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def csi_triplet(y_true: np.ndarray, y_pred: np.ndarray, threshold_mm: float) -> dict[str, float | int]:
    obs = y_true >= threshold_mm
    pred = y_pred >= threshold_mm
    tp = int(np.sum(obs & pred))
    fp = int(np.sum(~obs & pred))
    fn = int(np.sum(obs & ~pred))
    tn = int(np.sum(~obs & ~pred))
    pod = float(tp / (tp + fn)) if tp + fn else np.nan
    far = float(fp / (tp + fp)) if tp + fp else np.nan
    csi = float(tp / (tp + fp + fn)) if tp + fp + fn else np.nan
    acc = float((tp + tn) / (tp + fp + fn + tn)) if tp + fp + fn + tn else np.nan
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pod": pod,
        "far": far,
        "csi": csi,
        "accuracy": acc,
    }


def bootstrap_csi_difference(
    y_true: np.ndarray,
    pred_left: np.ndarray,
    pred_right: np.ndarray,
    block_codes: np.ndarray,
    threshold_mm: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    mask = np.isfinite(y_true) & np.isfinite(pred_left) & np.isfinite(pred_right)
    if not np.any(mask):
        return {
            "n_eval": 0,
            "n_blocks": 0,
            "csi_left": None,
            "csi_right": None,
            "delta_csi_left_minus_right": None,
            "ci_low": None,
            "ci_high": None,
            "prob_left_better": None,
            "ci_excludes_zero": None,
            "p_two_sided_boot": None,
        }

    y = y_true[mask].astype(np.float64, copy=False)
    left = pred_left[mask].astype(np.float64, copy=False)
    right = pred_right[mask].astype(np.float64, copy=False)
    codes = block_codes[mask].astype(np.int32, copy=False)

    n_blocks = int(codes.max()) + 1
    count = np.bincount(codes, minlength=n_blocks).astype(np.float64, copy=False)
    valid_blocks = count > 0

    obs = y >= threshold_mm
    left_event = left >= threshold_mm
    right_event = right >= threshold_mm

    tp_left = np.bincount(codes, weights=(obs & left_event).astype(np.float64), minlength=n_blocks)[valid_blocks]
    fp_left = np.bincount(codes, weights=((~obs) & left_event).astype(np.float64), minlength=n_blocks)[valid_blocks]
    fn_left = np.bincount(codes, weights=(obs & (~left_event)).astype(np.float64), minlength=n_blocks)[valid_blocks]

    tp_right = np.bincount(codes, weights=(obs & right_event).astype(np.float64), minlength=n_blocks)[valid_blocks]
    fp_right = np.bincount(codes, weights=((~obs) & right_event).astype(np.float64), minlength=n_blocks)[valid_blocks]
    fn_right = np.bincount(codes, weights=(obs & (~right_event)).astype(np.float64), minlength=n_blocks)[valid_blocks]

    def csi_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
        denom = tp + fp + fn
        out = np.full(denom.shape, np.nan, dtype=np.float64)
        np.divide(tp, denom, out=out, where=denom > 0.0)
        return out

    csi_left = csi_from_counts(tp_left.sum(keepdims=True), fp_left.sum(keepdims=True), fn_left.sum(keepdims=True))[0]
    csi_right = csi_from_counts(tp_right.sum(keepdims=True), fp_right.sum(keepdims=True), fn_right.sum(keepdims=True))[0]

    n_sample_blocks = int(valid_blocks.sum())
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(
        0,
        n_sample_blocks,
        size=(n_bootstrap, n_sample_blocks),
        endpoint=False,
        dtype=np.int32,
    )
    boot_csi_left = csi_from_counts(
        tp_left[sample_idx].sum(axis=1),
        fp_left[sample_idx].sum(axis=1),
        fn_left[sample_idx].sum(axis=1),
    )
    boot_csi_right = csi_from_counts(
        tp_right[sample_idx].sum(axis=1),
        fp_right[sample_idx].sum(axis=1),
        fn_right[sample_idx].sum(axis=1),
    )
    boot_diff = boot_csi_left - boot_csi_right
    finite_boot = np.isfinite(boot_diff)
    if not np.any(finite_boot):
        return {
            "n_eval": int(mask.sum()),
            "n_blocks": n_sample_blocks,
            "csi_left": float(csi_left),
            "csi_right": float(csi_right),
            "delta_csi_left_minus_right": float(csi_left - csi_right),
            "ci_low": None,
            "ci_high": None,
            "prob_left_better": None,
            "ci_excludes_zero": None,
            "p_two_sided_boot": None,
        }

    ci_low, ci_high = np.quantile(boot_diff[finite_boot], [0.025, 0.975])
    prob_left_better = float(np.mean(boot_diff[finite_boot] > 0.0))
    ci_excludes_zero = bool((ci_low > 0.0) or (ci_high < 0.0))
    p_two_sided = float(2.0 * min(np.mean(boot_diff[finite_boot] <= 0.0), np.mean(boot_diff[finite_boot] >= 0.0)))

    return {
        "n_eval": int(mask.sum()),
        "n_blocks": n_sample_blocks,
        "csi_left": float(csi_left),
        "csi_right": float(csi_right),
        "delta_csi_left_minus_right": float(csi_left - csi_right),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "prob_left_better": prob_left_better,
        "ci_excludes_zero": ci_excludes_zero,
        "p_two_sided_boot": p_two_sided,
    }


def train_conflict_policy(train_df: pd.DataFrame) -> dict[str, object]:
    conflict = train_df["observable_conflict_nonref_v0"].to_numpy(dtype=np.float64, copy=False)
    q75 = float(np.quantile(conflict[np.isfinite(conflict)], 0.75))
    robust = train_df["avamet_n_active"].to_numpy(dtype=np.int16, copy=False) >= 2
    low_conflict = robust & (conflict <= q75)
    high_conflict = robust & (conflict > q75)
    y_true = train_df["avamet_agg_mm"].to_numpy(dtype=np.float64, copy=False)

    low_candidates = {
        "IMERG": train_df["imerg_mm"].to_numpy(dtype=np.float64, copy=False),
        "ERA5": train_df["era5_mm"].to_numpy(dtype=np.float64, copy=False),
        "EURADCLIM": train_df["euradclim_on_imerg_mm"].to_numpy(dtype=np.float64, copy=False),
        LOCAL_LABEL: train_df[base.FIXED_LOCAL_BASELINE].to_numpy(dtype=np.float64, copy=False),
    }
    high_candidates = low_candidates
    low_choice = min(low_candidates, key=lambda name: mae(y_true[low_conflict], low_candidates[name][low_conflict]))
    high_choice = min(high_candidates, key=lambda name: mae(y_true[high_conflict], high_candidates[name][high_conflict]))
    return {
        "train_year": 2022,
        "conflict_threshold_q75_mm": q75,
        "low_conflict_choice": low_choice,
        "high_conflict_choice": high_choice,
    }


def build_policy_predictions(df: pd.DataFrame, policy: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    conflict = df["observable_conflict_nonref_v0"].to_numpy(dtype=np.float64, copy=False)
    threshold = float(policy["conflict_threshold_q75_mm"])
    low_mask = conflict <= threshold

    candidate_columns = {
        "IMERG": "imerg_mm",
        "ERA5": "era5_mm",
        "EURADCLIM": "euradclim_on_imerg_mm",
        LOCAL_LABEL: base.FIXED_LOCAL_BASELINE,
    }
    pred = np.full(df.shape[0], np.nan, dtype=np.float64)
    local_used = np.zeros(df.shape[0], dtype=bool)

    low_col = candidate_columns[str(policy["low_conflict_choice"])]
    high_col = candidate_columns[str(policy["high_conflict_choice"])]
    pred[low_mask] = df.loc[low_mask, low_col].to_numpy(dtype=np.float64, copy=False)
    pred[~low_mask] = df.loc[~low_mask, high_col].to_numpy(dtype=np.float64, copy=False)
    local_used[low_mask] = str(policy["low_conflict_choice"]) == LOCAL_LABEL
    local_used[~low_mask] = str(policy["high_conflict_choice"]) == LOCAL_LABEL
    return pred, local_used


def build_threshold_sweep(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    y_test = test_df["avamet_agg_mm"].to_numpy(dtype=np.float64, copy=False)
    robust = test_df["avamet_n_active"].to_numpy(dtype=np.int16, copy=False) >= 2
    wet = robust & (y_test > 0.0)
    test_conflict = test_df["observable_conflict_nonref_v0"].to_numpy(dtype=np.float64, copy=False)
    imerg_test = test_df["imerg_mm"].to_numpy(dtype=np.float64, copy=False)
    local_test = test_df[base.FIXED_LOCAL_BASELINE].to_numpy(dtype=np.float64, copy=False)

    train_conflict = train_df["observable_conflict_nonref_v0"].to_numpy(dtype=np.float64, copy=False)
    rows: list[dict[str, object]] = []
    for quantile in THRESHOLD_QUANTILES:
        threshold = float(np.quantile(train_conflict[np.isfinite(train_conflict)], quantile))
        use_imerg = test_conflict <= threshold
        pred = np.where(use_imerg, imerg_test, local_test)
        rows.append(
            {
                "train_quantile": quantile,
                "conflict_threshold_mm": threshold,
                "overall_mae_mm": mae(y_test[robust], pred[robust]),
                "positive_rain_mae_mm": mae(y_test[wet], pred[wet]),
                "local_usage_share": float(np.mean((~use_imerg)[robust])),
            }
        )
    return pd.DataFrame(rows).sort_values("train_quantile").reset_index(drop=True)


def build_simple_policy_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame, policy: dict[str, object]) -> pd.DataFrame:
    robust = test_df["avamet_n_active"].to_numpy(dtype=np.int16, copy=False) >= 2
    y_true = test_df["avamet_agg_mm"].to_numpy(dtype=np.float64, copy=False)
    wet = robust & (y_true > 0.0)
    train_conflict = train_df["observable_conflict_nonref_v0"].to_numpy(dtype=np.float64, copy=False)
    test_conflict = test_df["observable_conflict_nonref_v0"].to_numpy(dtype=np.float64, copy=False)

    candidate_columns = {
        "IMERG": "imerg_mm",
        "ERA5": "era5_mm",
        "EURADCLIM": "euradclim_on_imerg_mm",
        LOCAL_LABEL: base.FIXED_LOCAL_BASELINE,
    }

    def evaluate_variant(label: str, threshold: float, low_choice: str, high_choice: str) -> dict[str, object]:
        use_low = test_conflict <= threshold
        pred = np.where(
            use_low,
            test_df[candidate_columns[low_choice]].to_numpy(dtype=np.float64, copy=False),
            test_df[candidate_columns[high_choice]].to_numpy(dtype=np.float64, copy=False),
        )
        local_used = np.where(use_low, low_choice == LOCAL_LABEL, high_choice == LOCAL_LABEL)
        return {
            "model": label,
            "threshold_label": f"{threshold:.6f}",
            "low_conflict_choice": low_choice,
            "high_conflict_choice": high_choice,
            "overall_mae_mm": mae(y_true[robust], pred[robust]),
            "positive_rain_mae_mm": mae(y_true[wet], pred[wet]),
            "local_usage_share": float(np.mean(local_used[robust])),
        }

    q50 = float(np.quantile(train_conflict[np.isfinite(train_conflict)], 0.50))
    q75 = float(policy["conflict_threshold_q75_mm"])
    rows = [
        evaluate_variant(MEDIAN_GATE_LABEL, q50, "IMERG", LOCAL_LABEL),
        evaluate_variant(ALT_Q75_LABEL, q75, "EURADCLIM", LOCAL_LABEL),
    ]
    return pd.DataFrame(rows)


def evaluate_models(test_df: pd.DataFrame, policy: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    robust = test_df["avamet_n_active"].to_numpy(dtype=np.int16, copy=False) >= 2
    y_true = test_df["avamet_agg_mm"].to_numpy(dtype=np.float64, copy=False)
    wet = robust & (y_true > 0.0)
    block_codes, _ = factorize_blocks(test_df["time_utc"].to_numpy())

    policy_pred, local_used = build_policy_predictions(test_df, policy)
    model_predictions = {
        POLICY_LABEL: policy_pred,
        BEST_FIXED_EO_LABEL: test_df["euradclim_on_imerg_mm"].to_numpy(dtype=np.float64, copy=False),
        "ERA5": test_df["era5_mm"].to_numpy(dtype=np.float64, copy=False),
        "IMERG": test_df["imerg_mm"].to_numpy(dtype=np.float64, copy=False),
        LOCAL_LABEL: test_df[base.FIXED_LOCAL_BASELINE].to_numpy(dtype=np.float64, copy=False),
    }

    performance_rows: list[dict[str, object]] = []
    for model_name, pred in model_predictions.items():
        performance_rows.append(
            {
                "model": model_name,
                "overall_mae_mm": mae(y_true[robust], pred[robust]),
                "positive_rain_mae_mm": mae(y_true[wet], pred[wet]),
                "local_usage_share": float(np.mean(local_used[robust])) if model_name == POLICY_LABEL else (1.0 if model_name == LOCAL_LABEL else 0.0),
            }
        )

    performance_df = pd.DataFrame(performance_rows)

    comparisons = [
        (POLICY_LABEL, BEST_FIXED_EO_LABEL),
        (POLICY_LABEL, LOCAL_LABEL),
    ]
    bootstrap_rows: list[dict[str, object]] = []
    for left_name, right_name in comparisons:
        left_pred = model_predictions[left_name]
        right_pred = model_predictions[right_name]
        for slice_name, mask in [("overall", robust), ("positive_rain", wet)]:
            result = bootstrap_mae_difference(
                y_true=y_true[mask],
                pred_eo=left_pred[mask],
                pred_local=right_pred[mask],
                block_codes=block_codes[mask],
                n_bootstrap=N_BOOTSTRAP,
                seed=SEED,
            )
            bootstrap_rows.append(
                {
                    "comparison": f"{left_name} minus {right_name}",
                    "slice": slice_name,
                    **result,
                }
            )
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    return performance_df, bootstrap_df


def evaluate_alerts(test_df: pd.DataFrame, policy: dict[str, object]) -> pd.DataFrame:
    robust = test_df["avamet_n_active"].to_numpy(dtype=np.int16, copy=False) >= 2
    y_true = test_df["avamet_agg_mm"].to_numpy(dtype=np.float64, copy=False)[robust]
    policy_pred, _ = build_policy_predictions(test_df, policy)
    model_predictions = {
        POLICY_LABEL: policy_pred[robust],
        BEST_FIXED_EO_LABEL: test_df.loc[robust, "euradclim_on_imerg_mm"].to_numpy(dtype=np.float64, copy=False),
        "ERA5": test_df.loc[robust, "era5_mm"].to_numpy(dtype=np.float64, copy=False),
        "IMERG": test_df.loc[robust, "imerg_mm"].to_numpy(dtype=np.float64, copy=False),
        LOCAL_LABEL: test_df.loc[robust, base.FIXED_LOCAL_BASELINE].to_numpy(dtype=np.float64, copy=False),
    }
    rows: list[dict[str, object]] = []
    for threshold in ALERT_THRESHOLDS_MM:
        for model_name, pred in model_predictions.items():
            metrics = csi_triplet(y_true, pred, threshold)
            rows.append(
                {
                    "threshold_mm": threshold,
                    "model": model_name,
                    **metrics,
                }
            )
    return pd.DataFrame(rows).sort_values(["threshold_mm", "model"]).reset_index(drop=True)


def evaluate_alert_bootstrap(
    test_df: pd.DataFrame,
    policy: dict[str, object],
    block_unit: str = "hour",
) -> pd.DataFrame:
    robust = test_df["avamet_n_active"].to_numpy(dtype=np.int16, copy=False) >= 2
    y_true = test_df["avamet_agg_mm"].to_numpy(dtype=np.float64, copy=False)
    block_codes, _ = factorize_blocks(test_df["time_utc"].to_numpy(), unit=block_unit)
    policy_pred, _ = build_policy_predictions(test_df, policy)
    model_predictions = {
        POLICY_LABEL: policy_pred,
        BEST_FIXED_EO_LABEL: test_df["euradclim_on_imerg_mm"].to_numpy(dtype=np.float64, copy=False),
        LOCAL_LABEL: test_df[base.FIXED_LOCAL_BASELINE].to_numpy(dtype=np.float64, copy=False),
    }

    rows: list[dict[str, object]] = []
    comparisons = [
        (POLICY_LABEL, BEST_FIXED_EO_LABEL),
        (POLICY_LABEL, LOCAL_LABEL),
    ]
    for comp_idx, (left_name, right_name) in enumerate(comparisons):
        for threshold_idx, threshold in enumerate(ALERT_THRESHOLDS_MM):
            result = bootstrap_csi_difference(
                y_true=y_true[robust],
                pred_left=model_predictions[left_name][robust],
                pred_right=model_predictions[right_name][robust],
                block_codes=block_codes[robust],
                threshold_mm=threshold,
                n_bootstrap=N_BOOTSTRAP,
                seed=SEED + 100 * comp_idx + threshold_idx,
            )
            rows.append(
                {
                    "block_unit": block_unit,
                    "comparison": f"{left_name} minus {right_name}",
                    "threshold_mm": threshold,
                    **result,
                }
            )
    return pd.DataFrame(rows).sort_values(["comparison", "threshold_mm"]).reset_index(drop=True)


def plot_policy_figure(performance_df: pd.DataFrame, alerts_df: pd.DataFrame, threshold_sweep_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    perf_plot = performance_df.set_index("model").loc[[POLICY_LABEL, BEST_FIXED_EO_LABEL, LOCAL_LABEL]].reset_index()
    x = np.arange(perf_plot.shape[0])
    axes[0].bar(
        x - 0.16,
        perf_plot["overall_mae_mm"],
        width=0.32,
        color=[COLORS[name] for name in perf_plot["model"]],
        alpha=0.9,
        label="Overall MAE",
    )
    axes[0].bar(
        x + 0.16,
        perf_plot["positive_rain_mae_mm"],
        width=0.32,
        color=[COLORS[name] for name in perf_plot["model"]],
        alpha=0.45,
        label="Positive-rain MAE",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(perf_plot["model"], rotation=12)
    axes[0].set_title("MAE comparison on 2023")
    axes[0].set_ylabel("MAE (mm)")
    axes[0].legend(frameon=False)

    csi_plot = alerts_df[alerts_df["model"].isin([POLICY_LABEL, BEST_FIXED_EO_LABEL, LOCAL_LABEL])].copy()
    for model_name in [POLICY_LABEL, BEST_FIXED_EO_LABEL, LOCAL_LABEL]:
        sub = csi_plot[csi_plot["model"] == model_name]
        axes[1].plot(
            sub["threshold_mm"],
            sub["csi"],
            marker="o",
            linewidth=2.0,
            color=COLORS[model_name],
            label=model_name,
        )
    axes[1].set_xscale("log")
    axes[1].set_xticks(ALERT_THRESHOLDS_MM)
    axes[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[1].set_title("Alert CSI by rainfall threshold")
    axes[1].set_xlabel("Threshold (mm)")
    axes[1].set_ylabel("CSI")
    axes[1].legend(frameon=False)

    axes[2].plot(
        100.0 * threshold_sweep_df["local_usage_share"],
        threshold_sweep_df["overall_mae_mm"],
        color=COLORS[POLICY_LABEL],
        marker="o",
        linewidth=2.0,
    )
    for _, row in threshold_sweep_df.iterrows():
        label = f"q{int(round(100 * row['train_quantile']))}"
        axes[2].annotate(label, (100.0 * row["local_usage_share"], row["overall_mae_mm"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    axes[2].set_title("Conflict-gate trade-off")
    axes[2].set_xlabel("Local usage share on 2023 (%)")
    axes[2].set_ylabel("Overall MAE (mm)")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_conflict_policy_extension_v1.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure_conflict_policy_extension_v1.pdf", bbox_inches="tight")
    plt.close(fig)


def build_notes(
    policy: dict[str, object],
    performance_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    alert_bootstrap_df: pd.DataFrame,
    simple_baselines_df: pd.DataFrame,
) -> str:
    perf = performance_df.set_index("model")
    lines = [
        "# Conflict Policy Extension v1",
        "",
        "- Purpose: test whether observable gridded-product conflict can support a deployable source-selection rule trained on 2022 and evaluated on 2023.",
        f"- Learned threshold from 2022 q75 conflict: {fmt(policy['conflict_threshold_q75_mm'], 6)} mm.",
        f"- Learned low-conflict action: {policy['low_conflict_choice']}.",
        f"- Learned high-conflict action: {policy['high_conflict_choice']}.",
        "",
        "## 2023 MAE summary",
        f"- {POLICY_LABEL}: overall MAE = {fmt(perf.loc[POLICY_LABEL, 'overall_mae_mm'])}, positive-rain MAE = {fmt(perf.loc[POLICY_LABEL, 'positive_rain_mae_mm'])}, local usage share = {fmt(100.0 * perf.loc[POLICY_LABEL, 'local_usage_share'], 1)}%.",
        f"- {BEST_FIXED_EO_LABEL}: overall MAE = {fmt(perf.loc[BEST_FIXED_EO_LABEL, 'overall_mae_mm'])}, positive-rain MAE = {fmt(perf.loc[BEST_FIXED_EO_LABEL, 'positive_rain_mae_mm'])}.",
        f"- {LOCAL_LABEL}: overall MAE = {fmt(perf.loc[LOCAL_LABEL, 'overall_mae_mm'])}, positive-rain MAE = {fmt(perf.loc[LOCAL_LABEL, 'positive_rain_mae_mm'])}.",
        "",
        "## Simple routing baselines on 2023",
    ]
    for _, row in simple_baselines_df.iterrows():
        lines.append(
            f"- {row['model']}: overall MAE = {fmt(row['overall_mae_mm'])}, "
            f"positive-rain MAE = {fmt(row['positive_rain_mae_mm'])}, "
            f"local usage share = {fmt(100.0 * row['local_usage_share'], 1)}%."
        )
    lines.extend(
        [
            "",
        "## Bootstrap deltas on 2023",
        ]
    )
    for _, row in bootstrap_df.iterrows():
        lines.append(
            f"- {row['comparison']} / {row['slice']}: delta MAE = {fmt(row['delta_mae_eo_minus_local'])} "
            f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]."
        )
    lines.extend(["", "## Alert CSI on 2023"])
    for threshold in ALERT_THRESHOLDS_MM:
        subset = alerts_df[alerts_df["threshold_mm"] == threshold].set_index("model")
        lines.append(
            f"- Threshold {fmt(threshold, 1)} mm: "
            f"{POLICY_LABEL} CSI = {fmt(subset.loc[POLICY_LABEL, 'csi'], 3)}, "
            f"{BEST_FIXED_EO_LABEL} CSI = {fmt(subset.loc[BEST_FIXED_EO_LABEL, 'csi'], 3)}, "
            f"{LOCAL_LABEL} CSI = {fmt(subset.loc[LOCAL_LABEL, 'csi'], 3)}."
        )
    lines.extend(["", "## Alert CSI bootstrap contrasts on 2023"])
    for _, row in alert_bootstrap_df.iterrows():
        lines.append(
            f"- {row['comparison']} / {fmt(row['threshold_mm'], 1)} mm: delta CSI = {fmt(row['delta_csi_left_minus_right'], 4)} "
            f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = build_year_frame(2022)
    test_df = build_year_frame(2023)
    policy = train_conflict_policy(train_df)
    threshold_sweep_df = build_threshold_sweep(train_df, test_df)
    performance_df, bootstrap_df = evaluate_models(test_df, policy)
    alerts_df = evaluate_alerts(test_df, policy)
    alert_bootstrap_df = evaluate_alert_bootstrap(test_df, policy)
    simple_baselines_df = build_simple_policy_baselines(train_df, test_df, policy)

    threshold_sweep_df.to_csv(OUT_DIR / "conflict_threshold_sweep_v1.csv", index=False)
    performance_df.to_csv(OUT_DIR / "conflict_policy_performance_v1.csv", index=False)
    bootstrap_df.to_csv(OUT_DIR / "conflict_policy_bootstrap_v1.csv", index=False)
    alerts_df.to_csv(OUT_DIR / "conflict_policy_alert_metrics_v1.csv", index=False)
    alert_bootstrap_df.to_csv(OUT_DIR / "conflict_policy_alert_bootstrap_v1.csv", index=False)
    simple_baselines_df.to_csv(OUT_DIR / "conflict_policy_simple_baselines_v1.csv", index=False)
    (OUT_DIR / "conflict_policy_definition_v1.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    (OUT_DIR / "conflict_policy_notes_v1.md").write_text(
        build_notes(policy, performance_df, bootstrap_df, alerts_df, alert_bootstrap_df, simple_baselines_df),
        encoding="utf-8",
    )

    plot_policy_figure(performance_df, alerts_df, threshold_sweep_df)
    (OUT_DIR / "conflict_policy_manifest_v1.json").write_text(
        json.dumps(
            {
                "train_year": 2022,
                "test_year": 2023,
                "seed": SEED,
                "n_bootstrap": N_BOOTSTRAP,
                "fixed_local_baseline": base.FIXED_LOCAL_BASELINE,
                "alert_thresholds_mm": ALERT_THRESHOLDS_MM,
                "threshold_quantiles": THRESHOLD_QUANTILES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
