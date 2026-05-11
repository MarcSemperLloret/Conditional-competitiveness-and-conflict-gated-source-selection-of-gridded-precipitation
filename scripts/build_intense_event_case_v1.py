#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import build_conflict_policy_extension_v1 as ext
import build_delta_mae_by_support_score_and_regime_v1 as base
from build_temporal_sensitivity_2022_v1 import build_year_frame


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "intense_event_case_v1"
STATIC_PARQUET = base.STATIC_PARQUET

EVENT_GRID_ID = "imerg_1802_1304"
EVENT_START = pd.Timestamp("2023-09-02 00:00:00")
EVENT_END = pd.Timestamp("2023-09-03 23:00:00")
PEAK_HOUR = pd.Timestamp("2023-09-03 10:00:00")
PEAK_TOP_N = 5

COLORS = {
    "Target": "#111827",
    "Policy": "#1f6f8b",
    "Fixed local": "#6b7280",
    "EURADCLIM": "#2f8d62",
    "ERA5": "#e18c31",
    "IMERG": "#1677b3",
    "Conflict": "#7c3aed",
    "Peak": "#fee2e2",
    "Local window": "#ecfeff",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Aptos", "Segoe UI", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
        "savefig.dpi": 400,
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


def build_event_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    train_df = build_year_frame(2022)
    test_df = build_year_frame(2023)
    policy = ext.train_conflict_policy(train_df)
    policy_pred, policy_local_used = ext.build_policy_predictions(test_df, policy)

    event_mask = (test_df["grid_id"] == EVENT_GRID_ID) & (test_df["time_utc"] >= EVENT_START) & (test_df["time_utc"] <= EVENT_END)
    event_df = test_df.loc[
        event_mask,
        [
            "time_utc",
            "grid_id",
            "avamet_agg_mm",
            "observable_conflict_nonref_v0",
            "imerg_mm",
            "era5_mm",
            "euradclim_on_imerg_mm",
            base.FIXED_LOCAL_BASELINE,
            "avamet_n_active",
            "distance_to_coast_km",
            "elevation_m",
        ],
    ].copy()
    event_df["policy_pred"] = policy_pred[event_mask]
    event_df["policy_local_used"] = policy_local_used[event_mask]

    for label, column in [
        ("imerg_abs_err", "imerg_mm"),
        ("era5_abs_err", "era5_mm"),
        ("eurad_abs_err", "euradclim_on_imerg_mm"),
        ("local_abs_err", base.FIXED_LOCAL_BASELINE),
        ("policy_abs_err", "policy_pred"),
    ]:
        event_df[label] = (event_df[column] - event_df["avamet_agg_mm"]).abs()

    robust_peak = (
        (test_df["time_utc"] == PEAK_HOUR)
        & (test_df["avamet_n_active"].to_numpy(dtype=np.int16, copy=False) >= 2)
        & (test_df["avamet_agg_mm"] >= 20.0)
    )
    peak_df = test_df.loc[
        robust_peak,
        [
            "time_utc",
            "grid_id",
            "avamet_agg_mm",
            "observable_conflict_nonref_v0",
            "distance_to_coast_km",
            "elevation_m",
            "imerg_mm",
            "era5_mm",
            "euradclim_on_imerg_mm",
            base.FIXED_LOCAL_BASELINE,
        ],
    ].copy()
    peak_df["policy_pred"] = policy_pred[robust_peak]
    peak_df["policy_local_used"] = policy_local_used[robust_peak]
    peak_df["policy_branch"] = np.where(peak_df["policy_local_used"], "Fixed local", "IMERG")
    for label, column in [
        ("imerg_abs_err", "imerg_mm"),
        ("era5_abs_err", "era5_mm"),
        ("eurad_abs_err", "euradclim_on_imerg_mm"),
        ("local_abs_err", base.FIXED_LOCAL_BASELINE),
        ("policy_abs_err", "policy_pred"),
    ]:
        peak_df[label] = (peak_df[column] - peak_df["avamet_agg_mm"]).abs()
    peak_df = peak_df.sort_values("avamet_agg_mm", ascending=False).head(PEAK_TOP_N).reset_index(drop=True)
    peak_df.insert(0, "case_id", [f"C{i + 1}" for i in range(peak_df.shape[0])])
    peak_df["target_mm_label"] = peak_df["avamet_agg_mm"].map(lambda x: f"{x:.1f}")

    static_df = pd.read_parquet(STATIC_PARQUET)
    static_df = static_df[["grid_id", "lon", "lat"]].drop_duplicates()
    peak_df = peak_df.merge(static_df, on="grid_id", how="left")

    event_meta = {
        "event_grid_id": EVENT_GRID_ID,
        "event_start": str(EVENT_START),
        "event_end": str(EVENT_END),
        "peak_hour": str(PEAK_HOUR),
        "policy_threshold": float(policy["conflict_threshold_q75_mm"]),
        "event_cell_distance_to_coast_km": float(event_df["distance_to_coast_km"].iloc[0]),
        "event_cell_elevation_m": float(event_df["elevation_m"].iloc[0]),
        "peak_event_max_target_mm": float(peak_df["avamet_agg_mm"].max()),
        "peak_event_n_cells_ge20mm": int(peak_df.shape[0]),
    }
    return event_df.reset_index(drop=True), peak_df, event_meta


def plot_selected_cell_case(event_df: pd.DataFrame, policy_threshold: float) -> None:
    fig = plt.figure(figsize=(12.4, 7.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.15], hspace=0.22)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_conflict = fig.add_subplot(gs[1, 0], sharex=ax_top)

    times = pd.to_datetime(event_df["time_utc"])
    peak_start = pd.Timestamp("2023-09-03 08:00:00")
    peak_end = pd.Timestamp("2023-09-03 11:00:00")

    ax_top.axvspan(peak_start, peak_end, color=COLORS["Peak"], alpha=0.45, zorder=0)
    ax_top.plot(times, event_df["avamet_agg_mm"], color=COLORS["Target"], linewidth=2.6, marker="o", markersize=3.8, label="Target")
    ax_top.plot(times, event_df["policy_pred"], color=COLORS["Policy"], linewidth=2.4, label="Policy")
    ax_top.plot(times, event_df[base.FIXED_LOCAL_BASELINE], color=COLORS["Fixed local"], linewidth=1.8, linestyle="--", label="Fixed local")
    ax_top.plot(times, event_df["euradclim_on_imerg_mm"], color=COLORS["EURADCLIM"], linewidth=1.8, label="EURADCLIM")
    ax_top.plot(times, event_df["era5_mm"], color=COLORS["ERA5"], linewidth=1.4, label="ERA5")
    ax_top.plot(times, event_df["imerg_mm"], color=COLORS["IMERG"], linewidth=1.4, label="IMERG")
    ax_top.set_ylabel("Hourly rainfall (mm)")
    ax_top.set_title("A. Selected-cell rainfall traces during the intense episode", loc="left", pad=10)
    ax_top.grid(axis="y", color="#d9e1ea", linewidth=0.8, alpha=0.9)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.legend(frameon=False, ncol=3, loc="upper left")
    peak_rows = event_df[event_df["avamet_agg_mm"] >= 40.0]
    for _, row in peak_rows.iterrows():
        ax_top.annotate(
            f"{row['avamet_agg_mm']:.1f}",
            (row["time_utc"], row["avamet_agg_mm"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        fontsize=9,
        color=COLORS["Target"],
    )

    ax_conflict.axvspan(peak_start, peak_end, color=COLORS["Peak"], alpha=0.45, zorder=0)
    local_mask = event_df["policy_local_used"].to_numpy(dtype=bool, copy=False)
    mask_edges = np.diff(np.r_[False, local_mask, False].astype(int))
    start_indices = np.where(mask_edges == 1)[0]
    end_indices = np.where(mask_edges == -1)[0]
    for start_idx, end_idx in zip(start_indices, end_indices, strict=True):
        start_time = times.iloc[start_idx]
        end_time = times.iloc[end_idx - 1] + pd.Timedelta(hours=1)
        ax_conflict.axvspan(start_time, end_time, color=COLORS["Local window"], alpha=0.65, zorder=0)
    ax_conflict.plot(times, event_df["observable_conflict_nonref_v0"], color=COLORS["Conflict"], linewidth=2.2, marker="o", markersize=3.0)
    ax_conflict.axhline(policy_threshold, color=COLORS["Policy"], linestyle="--", linewidth=1.4, label="Policy threshold")
    ax_conflict.set_ylabel("Conflict (mm)")
    ax_conflict.set_title("B. Observable conflict and branch selection", loc="left", pad=10)
    ax_conflict.grid(axis="y", color="#d9e1ea", linewidth=0.8, alpha=0.9)
    ax_conflict.spines["top"].set_visible(False)
    ax_conflict.spines["right"].set_visible(False)
    ax_conflict.text(
        0.01,
        0.96,
        "Shaded cyan: policy selects fixed local",
        transform=ax_conflict.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="#0f4c5c",
    )

    locator = mdates.HourLocator(interval=6)
    formatter = mdates.DateFormatter("%d %b\n%H:%M")
    ax_conflict.xaxis.set_major_locator(locator)
    ax_conflict.xaxis.set_major_formatter(formatter)
    ax_conflict.set_xlabel("UTC time")
    for label in ax_top.get_xticklabels():
        label.set_visible(False)

    fig.suptitle("Selected-cell view of the 3 September 2023 intense event", x=0.07, y=0.98, ha="left", fontsize=15, fontweight="bold")
    save_exports(fig, "figure_intense_event_selected_cell_v1")


def plot_peak_cells_case(peak_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.8))

    x = np.arange(peak_df.shape[0], dtype=float)
    width = 0.36
    ax.bar(x - width / 2.0, peak_df["eurad_abs_err"], width=width, color=COLORS["EURADCLIM"], label="EURADCLIM")
    ax.bar(x + width / 2.0, peak_df["policy_abs_err"], width=width, color=COLORS["Policy"], label="Policy-selected source")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cid}\n{target}" for cid, target in zip(peak_df["case_id"], peak_df["target_mm_label"], strict=True)])
    ax.set_ylabel("Absolute error (mm)")
    ax.set_xlabel("Peak-hour cells (labels show target mm)")
    ax.set_title("Peak hour at 10:00 UTC across the five strongest robust-support cells", loc="left", pad=10)
    ax.grid(axis="y", color="#d9e1ea", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    fig.suptitle("Peak-hour comparison for the 3 September 2023 event", x=0.07, y=0.98, ha="left", fontsize=15, fontweight="bold")
    save_exports(fig, "figure_intense_event_peak_cells_v1")


def build_notes(event_df: pd.DataFrame, peak_df: pd.DataFrame, meta: dict[str, object]) -> str:
    peak_hours = event_df[event_df["avamet_agg_mm"] >= 20.0].copy()
    lines = [
        "# Intense Event Case v1",
        "",
        "- Purpose: add a purely illustrative intense-event case that makes the operational source-selection logic easier to read.",
        f"- Selected cell: {meta['event_grid_id']} ({fmt(meta['event_cell_distance_to_coast_km'], 1)} km from coast; {fmt(meta['event_cell_elevation_m'], 0)} m elevation).",
        f"- Event window: {meta['event_start']} to {meta['event_end']}.",
        f"- Policy conflict threshold: {fmt(meta['policy_threshold'], 6)} mm.",
        "",
        "## Selected-cell peak hours",
    ]
    for _, row in peak_hours.iterrows():
        lines.append(
            f"- {row['time_utc']}: target = {fmt(row['avamet_agg_mm'], 1)} mm, "
            f"policy abs. error = {fmt(row['policy_abs_err'], 1)} mm, "
            f"EURADCLIM abs. error = {fmt(row['eurad_abs_err'], 1)} mm, "
            f"branch = {'Fixed local' if row['policy_local_used'] else 'IMERG'}."
        )
    lines.extend(["", "## Peak-hour top-five cells"])
    for _, row in peak_df.iterrows():
        lines.append(
            f"- {row['case_id']} / {row['grid_id']}: target = {fmt(row['avamet_agg_mm'], 1)} mm, "
            f"conflict = {fmt(row['observable_conflict_nonref_v0'], 3)} mm, "
            f"policy branch = {row['policy_branch']}, policy abs. error = {fmt(row['policy_abs_err'], 1)} mm, "
            f"EURADCLIM abs. error = {fmt(row['eurad_abs_err'], 1)} mm."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    event_df, peak_df, meta = build_event_frames()
    plot_selected_cell_case(event_df, float(meta["policy_threshold"]))
    plot_peak_cells_case(peak_df)

    event_path = OUT_DIR / "intense_event_selected_cell_timeseries_v1.csv"
    peak_path = OUT_DIR / "intense_event_peak_cells_v1.csv"
    notes_path = OUT_DIR / "intense_event_notes_v1.md"
    manifest_path = OUT_DIR / "intense_event_manifest_v1.json"

    event_df.to_csv(event_path, index=False)
    peak_df.to_csv(peak_path, index=False)
    notes_path.write_text(build_notes(event_df, peak_df, meta), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "event_meta": meta,
                "outputs": {
                    "intense_event_selected_cell_timeseries_v1.csv": str(event_path),
                    "intense_event_peak_cells_v1.csv": str(peak_path),
                    "figure_intense_event_selected_cell_v1.png": str(OUT_DIR / "figure_intense_event_selected_cell_v1.png"),
                    "figure_intense_event_selected_cell_v1.pdf": str(OUT_DIR / "figure_intense_event_selected_cell_v1.pdf"),
                    "figure_intense_event_peak_cells_v1.png": str(OUT_DIR / "figure_intense_event_peak_cells_v1.png"),
                    "figure_intense_event_peak_cells_v1.pdf": str(OUT_DIR / "figure_intense_event_peak_cells_v1.pdf"),
                    "intense_event_notes_v1.md": str(notes_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

