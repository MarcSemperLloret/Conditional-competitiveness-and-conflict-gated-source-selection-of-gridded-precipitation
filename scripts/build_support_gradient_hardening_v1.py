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
from run_delta_mae_block_bootstrap_v1 import bootstrap_mae_difference, factorize_blocks


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "support_gradient_hardening_v1"
YEAR = 2023
N_BOOTSTRAP = 1000
SEED = 42
FIG_DPI = 400

EO_LABELS = {
    "source_imerg": "IMERG",
    "source_era5": "ERA5",
    "source_euradclim": "EURADCLIM",
}
PRODUCT_ORDER = ["IMERG", "ERA5", "EURADCLIM"]
COLORS = {
    "IMERG": "#1677b3",
    "ERA5": "#e18c31",
    "EURADCLIM": "#2f8d62",
}
SLICE_ORDER = {
    "overall_all": 0,
    "overall_positive_target": 1,
    "conflict_low": 2,
    "conflict_mid": 3,
    "conflict_tail": 4,
}
SLICE_LABELS = {
    "overall_all": "Overall",
    "overall_positive_target": "Positive-rain only",
    "conflict_low": "Low conflict",
    "conflict_mid": "Mid conflict",
    "conflict_tail": "High conflict",
}
SUPPORT_FAMILY_LABELS = {
    "target_active_gauges": "Target-cell active gauges",
    "leavecell_active_15km": "Active leave-cell gauges within 15 km",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Aptos", "Segoe UI", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
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


def add_support_bins(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    out = df.copy()
    robust = out["avamet_n_active"].to_numpy(dtype=np.int16, copy=False) >= 2

    out["target_active_bin"] = np.select(
        [
            out["avamet_n_active"] == 2,
            out["avamet_n_active"] == 3,
            out["avamet_n_active"] == 4,
            out["avamet_n_active"] >= 5,
        ],
        ["2", "3", "4", "5+"],
        default="other",
    )

    q25 = int(out.loc[robust, "leavecell_active_15km"].quantile(0.25))
    q50 = int(out.loc[robust, "leavecell_active_15km"].quantile(0.50))
    q75 = int(out.loc[robust, "leavecell_active_15km"].quantile(0.75))
    external_labels = [f"0-{q25}", f"{q25 + 1}-{q50}", f"{q50 + 1}-{q75}", f"{q75 + 1}+"]
    out["leavecell_active_15km_bin"] = np.select(
        [
            out["leavecell_active_15km"] <= q25,
            (out["leavecell_active_15km"] > q25) & (out["leavecell_active_15km"] <= q50),
            (out["leavecell_active_15km"] > q50) & (out["leavecell_active_15km"] <= q75),
            out["leavecell_active_15km"] > q75,
        ],
        external_labels,
        default="other",
    )

    meta = {
        "external_support_thresholds": {
            "q25": q25,
            "q50": q50,
            "q75": q75,
            "labels": external_labels,
        }
    }
    return out, meta


def build_bootstrap_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    df, meta = add_support_bins(df)
    robust = df["avamet_n_active"].to_numpy(dtype=np.int16, copy=False) >= 2
    positive = df["avamet_agg_mm"].to_numpy(dtype=np.float32, copy=False) > 0.0
    conflict = df["observable_conflict_bin"].astype(str).to_numpy()
    y_true = df["avamet_agg_mm"].to_numpy(dtype=np.float32, copy=False)
    local = df[base.FIXED_LOCAL_BASELINE].to_numpy(dtype=np.float32, copy=False)
    block_codes, _ = factorize_blocks(df["time_utc"].to_numpy())

    slice_defs = [
        ("overall_all", robust),
        ("overall_positive_target", robust & positive),
        ("conflict_low", robust & (conflict == "low_le_p75")),
        ("conflict_mid", robust & (conflict == "mid_p75_p95")),
        ("conflict_tail", robust & (conflict == "tail_gt_p95")),
    ]
    support_defs = [
        ("target_active_gauges", "target_active_bin", ["2", "3", "4", "5+"]),
        (
            "leavecell_active_15km",
            "leavecell_active_15km_bin",
            list(meta["external_support_thresholds"]["labels"]),
        ),
    ]

    rows: list[dict[str, object]] = []
    for support_family, support_col, support_labels in support_defs:
        support_values = df[support_col].astype(str).to_numpy()
        for slice_id, slice_mask in slice_defs:
            for support_order, support_label in enumerate(support_labels):
                mask = slice_mask & (support_values == support_label)
                if not np.any(mask):
                    continue
                subset = df.loc[mask]
                for eo_name, eo_col in base.EO_COLUMNS.items():
                    result = bootstrap_mae_difference(
                        y_true=y_true[mask],
                        pred_eo=df[eo_col].to_numpy(dtype=np.float32, copy=False)[mask],
                        pred_local=local[mask],
                        block_codes=block_codes[mask],
                        n_bootstrap=N_BOOTSTRAP,
                        seed=SEED,
                    )
                    rows.append(
                        {
                            "year": YEAR,
                            "support_family": support_family,
                            "support_family_label": SUPPORT_FAMILY_LABELS[support_family],
                            "support_bin": support_label,
                            "support_order": support_order,
                            "slice_id": slice_id,
                            "slice_label": SLICE_LABELS[slice_id],
                            "slice_order": SLICE_ORDER[slice_id],
                            "eo_baseline": eo_name,
                            "eo_label": EO_LABELS[eo_name],
                            "n_rows": int(mask.sum()),
                            "n_hours": int(subset["time_utc"].nunique()),
                            "n_positive_target": int((subset["avamet_agg_mm"] > 0.0).sum()),
                            "mean_target_mm": float(subset["avamet_agg_mm"].mean()),
                            "mean_n_active": float(subset["avamet_n_active"].mean()),
                            "mean_leavecell_active_15km": float(subset["leavecell_active_15km"].mean()),
                            "mean_leavecell_nearest_active_km": float(subset["leavecell_nearest_active_km"].mean()),
                            "mean_leavecell_idw_neff": float(subset["leavecell_idw_neff"].mean()),
                            **result,
                        }
                    )

    out = pd.DataFrame(rows).sort_values(
        ["support_family", "slice_order", "support_order", "eo_label"]
    ).reset_index(drop=True)
    best = (
        out.loc[out.groupby(["support_family", "slice_id", "support_bin"])["delta_mae_eo_minus_local"].idxmin()]
        .copy()
        .rename(
            columns={
                "eo_label": "best_eo_label",
                "delta_mae_eo_minus_local": "best_eo_delta_mae_eo_minus_local",
                "ci_low": "best_eo_ci_low",
                "ci_high": "best_eo_ci_high",
            }
        )
    )
    return out, best, meta


def plot_support_gradient(bootstrap_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2), sharex="col")
    panel_specs = [
        ("overall_positive_target", "target_active_gauges", "Positive-rain rows by target-cell support"),
        ("overall_positive_target", "leavecell_active_15km", "Positive-rain rows by leave-cell external support"),
        ("conflict_tail", "target_active_gauges", "High-conflict rows by target-cell support"),
        ("conflict_tail", "leavecell_active_15km", "High-conflict rows by leave-cell external support"),
    ]
    offsets = {"IMERG": -0.18, "ERA5": 0.0, "EURADCLIM": 0.18}

    y_min = float(bootstrap_df["ci_low"].min())
    y_max = float(bootstrap_df["ci_high"].max())
    pad = 0.06 * max(abs(y_min), abs(y_max), 1.0)

    for ax, (slice_id, support_family, title) in zip(axes.flat, panel_specs, strict=False):
        sub = bootstrap_df[
            (bootstrap_df["slice_id"] == slice_id) & (bootstrap_df["support_family"] == support_family)
        ].copy()
        support_labels = sub.sort_values("support_order")["support_bin"].drop_duplicates().tolist()
        x = np.arange(len(support_labels), dtype=float)

        ax.axhline(0.0, color="#aab5c0", linewidth=1.1, linestyle="--", zorder=0)
        ax.grid(axis="y", color="#d9e1ea", linewidth=0.8, alpha=0.9)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", pad=10)

        for product in PRODUCT_ORDER:
            prod = sub[sub["eo_label"] == product].sort_values("support_order")
            y = prod["delta_mae_eo_minus_local"].to_numpy(dtype=float)
            yerr_low = y - prod["ci_low"].to_numpy(dtype=float)
            yerr_high = prod["ci_high"].to_numpy(dtype=float) - y
            ax.errorbar(
                x + offsets[product],
                y,
                yerr=np.vstack([yerr_low, yerr_high]),
                fmt="o-",
                color=COLORS[product],
                ecolor=COLORS[product],
                linewidth=1.7,
                elinewidth=1.4,
                capsize=3.0,
                markersize=5.4,
                markerfacecolor=COLORS[product],
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=product,
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(support_labels)
        ax.set_ylim(y_min - pad, y_max + pad)
        if support_family == "target_active_gauges":
            ax.set_ylabel("Delta MAE vs fixed local (mm)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False, ncol=3)
    fig.suptitle("Support-gradient sensitivity of gridded-minus-local MAE gaps", x=0.06, ha="left", y=1.05)
    save_exports(fig, "figure_support_gradient_hardening_v1")


def build_markdown(best_df: pd.DataFrame, meta: dict[str, object]) -> str:
    lines = [
        "# Support Gradient Hardening v1",
        "",
        f"- Year: `{YEAR}`",
        f"- Fixed local baseline: `{base.FIXED_LOCAL_BASELINE}`",
        "- Purpose: test whether the gridded-local gap changes gradually with stronger local support instead of behaving like a binary design artifact.",
        (
            "- External leave-cell support bins (robust rows only): "
            f"`{meta['external_support_thresholds']['labels'][0]}`, "
            f"`{meta['external_support_thresholds']['labels'][1]}`, "
            f"`{meta['external_support_thresholds']['labels'][2]}`, "
            f"`{meta['external_support_thresholds']['labels'][3]}` active gauges within `15 km`."
        ),
        "",
        "## Best-EO summary by support axis",
    ]

    for support_family in ["target_active_gauges", "leavecell_active_15km"]:
        lines.append(f"- `{SUPPORT_FAMILY_LABELS[support_family]}`")
        for slice_id in ["overall_positive_target", "conflict_tail"]:
            sub = best_df[
                (best_df["support_family"] == support_family) & (best_df["slice_id"] == slice_id)
            ].sort_values("support_order")
            if sub.empty:
                continue
            summary = "; ".join(
                (
                    f"{row['support_bin']}: {row['best_eo_label']} "
                    f"{fmt(row['best_eo_delta_mae_eo_minus_local'])} "
                    f"[{fmt(row['best_eo_ci_low'])}, {fmt(row['best_eo_ci_high'])}]"
                )
                for _, row in sub.iterrows()
            )
            lines.append(f"  - `{SLICE_LABELS[slice_id]}`: {summary}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_year_prediction_frame_with_support(YEAR)
    bootstrap_df, best_df, meta = build_bootstrap_table(df)

    csv_path = OUT_DIR / "support_gradient_bootstrap_v1.csv"
    best_path = OUT_DIR / "support_gradient_best_eo_v1.csv"
    md_path = OUT_DIR / "support_gradient_notes_v1.md"
    manifest_path = OUT_DIR / "support_gradient_manifest_v1.json"

    bootstrap_df.to_csv(csv_path, index=False)
    best_df.to_csv(best_path, index=False)
    plot_support_gradient(bootstrap_df)
    md_path.write_text(build_markdown(best_df, meta), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "year": YEAR,
                "fixed_local_baseline": base.FIXED_LOCAL_BASELINE,
                "bootstrap_resamples": N_BOOTSTRAP,
                "seed": SEED,
                "external_support_thresholds": meta["external_support_thresholds"],
                "bootstrap_csv": str(csv_path),
                "best_eo_csv": str(best_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

