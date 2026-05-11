#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_delta_mae_by_support_score_and_regime_v1 import (
    FIXED_LOCAL_BASELINE,
    LOCAL_BASELINES,
    EO_COLUMNS,
    build_prediction_frame,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "bootstrap_delta_mae_v1"
DEFAULT_N_BOOTSTRAP = 1500
DEFAULT_SEED = 42


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run hour-block bootstrap intervals for EO vs local delta MAE comparisons in paper_11."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--block-unit", choices=["hour", "day"], default="hour")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def fmt(value: object) -> str:
    if value is None:
        return "null"
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_f):
        return "null"
    return f"{value_f:.4f}"


def factorize_blocks(blocks: np.ndarray, unit: str = "hour") -> tuple[np.ndarray, np.ndarray]:
    block_index = pd.Index(pd.to_datetime(blocks))
    if unit == "day":
        block_index = block_index.floor("D")
    elif unit == "hour":
        block_index = block_index.floor("h")
    else:
        raise ValueError(f"Unsupported block unit: {unit}")
    codes, uniques = pd.factorize(block_index, sort=True)
    return codes.astype(np.int32, copy=False), uniques.to_numpy()


def bootstrap_mae_difference(
    y_true: np.ndarray,
    pred_eo: np.ndarray,
    pred_local: np.ndarray,
    block_codes: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    mask = np.isfinite(y_true) & np.isfinite(pred_eo) & np.isfinite(pred_local)
    if not np.any(mask):
        return {
            "n_eval": 0,
            "n_blocks": 0,
            "mae_eo": None,
            "mae_local": None,
            "delta_mae_eo_minus_local": None,
            "ci_low": None,
            "ci_high": None,
            "prob_eo_better": None,
            "ci_excludes_zero": None,
            "p_two_sided_boot": None,
        }

    y = y_true[mask].astype(np.float64, copy=False)
    eo = pred_eo[mask].astype(np.float64, copy=False)
    local = pred_local[mask].astype(np.float64, copy=False)
    codes = block_codes[mask].astype(np.int32, copy=False)

    n_blocks = int(codes.max()) + 1
    err_eo = np.abs(eo - y)
    err_local = np.abs(local - y)
    sum_eo = np.bincount(codes, weights=err_eo, minlength=n_blocks).astype(np.float64, copy=False)
    sum_local = np.bincount(codes, weights=err_local, minlength=n_blocks).astype(np.float64, copy=False)
    count = np.bincount(codes, minlength=n_blocks).astype(np.float64, copy=False)
    valid_blocks = count > 0
    sum_eo = sum_eo[valid_blocks]
    sum_local = sum_local[valid_blocks]
    count = count[valid_blocks]
    n_sample_blocks = int(count.size)

    mae_eo = float(sum_eo.sum() / count.sum())
    mae_local = float(sum_local.sum() / count.sum())
    delta = mae_eo - mae_local

    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(
        0,
        n_sample_blocks,
        size=(n_bootstrap, n_sample_blocks),
        endpoint=False,
        dtype=np.int32,
    )
    boot_sum_eo = sum_eo[sample_idx].sum(axis=1)
    boot_sum_local = sum_local[sample_idx].sum(axis=1)
    boot_count = count[sample_idx].sum(axis=1)
    boot_diff = boot_sum_eo / boot_count - boot_sum_local / boot_count

    ci_low, ci_high = np.quantile(boot_diff, [0.025, 0.975])
    prob_eo_better = float(np.mean(boot_diff < 0.0))
    ci_excludes_zero = bool((ci_low > 0.0) or (ci_high < 0.0))
    p_two_sided = float(2.0 * min(np.mean(boot_diff <= 0.0), np.mean(boot_diff >= 0.0)))

    return {
        "n_eval": int(mask.sum()),
        "n_blocks": n_sample_blocks,
        "mae_eo": mae_eo,
        "mae_local": mae_local,
        "delta_mae_eo_minus_local": float(delta),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "prob_eo_better": prob_eo_better,
        "ci_excludes_zero": ci_excludes_zero,
        "p_two_sided_boot": p_two_sided,
    }


def choose_best_local(y_true: np.ndarray, predictions: dict[str, np.ndarray], mask: np.ndarray) -> tuple[str, np.ndarray]:
    candidates: list[tuple[float, str]] = []
    for baseline in LOCAL_BASELINES:
        finite = np.isfinite(y_true[mask]) & np.isfinite(predictions[baseline][mask])
        if not np.any(finite):
            continue
        mae = float(np.mean(np.abs(predictions[baseline][mask][finite] - y_true[mask][finite])))
        candidates.append((mae, baseline))
    if not candidates:
        raise ValueError("No valid local baseline found for the requested slice.")
    _, best_name = min(candidates)
    return best_name, predictions[best_name]


def build_slice_masks(df: pd.DataFrame) -> list[dict[str, object]]:
    robust_mask = df["avamet_n_active"].to_numpy(dtype=np.int16, copy=False) >= 2
    positive_mask = robust_mask & (df["avamet_agg_mm"].to_numpy(dtype=np.float32, copy=False) > 0.0)
    conflict = df["observable_conflict_bin"].astype(str).to_numpy()

    slices = [
        {
            "slice_id": "overall_all",
            "slice_family": "overall",
            "slice_label": "all",
            "mask": robust_mask,
        },
        {
            "slice_id": "overall_positive_target",
            "slice_family": "overall",
            "slice_label": "positive_target_only",
            "mask": positive_mask,
        },
        {
            "slice_id": "conflict_low",
            "slice_family": "observable_conflict",
            "slice_label": "low_le_p75",
            "mask": robust_mask & (conflict == "low_le_p75"),
        },
        {
            "slice_id": "conflict_mid",
            "slice_family": "observable_conflict",
            "slice_label": "mid_p75_p95",
            "mask": robust_mask & (conflict == "mid_p75_p95"),
        },
        {
            "slice_id": "conflict_tail",
            "slice_family": "observable_conflict",
            "slice_label": "tail_gt_p95",
            "mask": robust_mask & (conflict == "tail_gt_p95"),
        },
    ]
    return slices


def build_report(summary_df: pd.DataFrame, n_bootstrap: int, block_unit: str) -> str:
    lines = [
        "# Delta MAE Block Bootstrap v1",
        "",
        f"- Bootstrap type: {block_unit}-block bootstrap over 2023 common-overlap timestamps.",
        "- Negative `delta_mae_eo_minus_local` means the EO product beats the local comparator on MAE.",
        f"- Resamples: {n_bootstrap}",
        "",
        "| Slice | EO | Comparator | Local baseline | n_eval | n_blocks | Î”(EO-local) | 95% CI | CI excludes 0 | P(EO better) |",
        "|------|----|------------|----------------|--------|----------|-------------|--------|---------------|--------------|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['slice_id']} | {row['eo_baseline']} | {row['comparator_kind']} | {row['local_baseline']} | "
            f"{int(row['n_eval'])} | {int(row['n_blocks'])} | {fmt(row['delta_mae_eo_minus_local'])} | "
            f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {bool(row['ci_excludes_zero'])} | {fmt(row['prob_eo_better'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary_csv": args.output_dir / "bootstrap_summary.csv",
        "report_md": args.output_dir / "bootstrap_report.md",
        "manifest_json": args.output_dir / "bootstrap_manifest.json",
    }
    if not args.overwrite:
        existing = [path for path in outputs.values() if path.exists()]
        if existing:
            raise FileExistsError(f"Outputs already exist. Re-run with --overwrite to replace them: {existing}")

    df = build_prediction_frame()
    y_true = df["avamet_agg_mm"].to_numpy(dtype=np.float32, copy=False)
    block_codes, block_values = factorize_blocks(df["time_utc"].to_numpy(), unit=args.block_unit)
    predictions = {
        eo_name: df[col].to_numpy(dtype=np.float32, copy=False)
        for eo_name, col in EO_COLUMNS.items()
    }
    for local_name in LOCAL_BASELINES:
        predictions[local_name] = df[local_name].to_numpy(dtype=np.float32, copy=False)

    rows: list[dict[str, object]] = []
    for slice_meta in build_slice_masks(df):
        slice_mask = slice_meta["mask"]
        if not np.any(slice_mask):
            continue

        best_local_name, best_local_pred = choose_best_local(y_true, predictions, slice_mask)
        comparator_defs = [
            {
                "comparator_kind": "fixed_local",
                "local_baseline": FIXED_LOCAL_BASELINE,
                "local_pred": predictions[FIXED_LOCAL_BASELINE],
            },
            {
                "comparator_kind": "best_local_in_slice",
                "local_baseline": best_local_name,
                "local_pred": best_local_pred,
            },
        ]

        for eo_name in EO_COLUMNS:
            for comparator in comparator_defs:
                result = bootstrap_mae_difference(
                    y_true=y_true[slice_mask],
                    pred_eo=predictions[eo_name][slice_mask],
                    pred_local=comparator["local_pred"][slice_mask],
                    block_codes=block_codes[slice_mask],
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed,
                )
                row = {
                    "slice_id": slice_meta["slice_id"],
                    "slice_family": slice_meta["slice_family"],
                    "slice_label": slice_meta["slice_label"],
                    "eo_baseline": eo_name,
                    "comparator_kind": comparator["comparator_kind"],
                    "local_baseline": comparator["local_baseline"],
                }
                row.update(result)
                rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values(
        ["slice_family", "slice_label", "eo_baseline", "comparator_kind"]
    ).reset_index(drop=True)
    summary_df.to_csv(outputs["summary_csv"], index=False)
    outputs["report_md"].write_text(build_report(summary_df, args.n_bootstrap, args.block_unit), encoding="utf-8")
    outputs["manifest_json"].write_text(
        json.dumps(
            {
                "n_bootstrap": int(args.n_bootstrap),
                "seed": int(args.seed),
                "block_unit": f"time_utc_{args.block_unit}",
                "scenario": "robust_ge_2",
                "local_baselines": LOCAL_BASELINES,
                "fixed_local_baseline": FIXED_LOCAL_BASELINE,
                "eo_baselines": list(EO_COLUMNS),
                "n_time_blocks": int(len(block_values)),
                "n_rows_summary": int(summary_df.shape[0]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

