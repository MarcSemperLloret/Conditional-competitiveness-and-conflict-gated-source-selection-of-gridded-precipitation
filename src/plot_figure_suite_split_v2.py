#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "paper_11_density_thresholds" / "results" / "main_figure_v1"
OUT_DIR = ROOT / "paper_11_density_thresholds" / "results" / "main_figure_v2_split"

PANEL_A_CSV = DATA_DIR / "panel_a_global_and_positive_rain.csv"
PANEL_B_CSV = DATA_DIR / "panel_b_conflict_bootstrap.csv"
PANEL_C_CSV = DATA_DIR / "panel_c_best_eo_by_regime.csv"

FIG_DPI = 400
BG = "#ffffff"
PANEL_BG = "#ffffff"
INK = "#1f2a33"
MUTED = "#677381"
GRID = "#d9e1ea"
ZERO = "#aab5c0"
IMERG = "#1677b3"
ERA5 = "#e18c31"
EURAD = "#2f8d62"

COLORS = {
    "IMERG": IMERG,
    "ERA5": ERA5,
    "EURADCLIM": EURAD,
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Aptos", "Segoe UI", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "axes.titleweight": "bold",
        "axes.facecolor": PANEL_BG,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
        "savefig.dpi": FIG_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def add_rounded_panel(ax: plt.Axes, pad: float = 0.01) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle=f"round,pad={pad},rounding_size=0.03",
            transform=ax.transAxes,
            facecolor=PANEL_BG,
            edgecolor=GRID,
            linewidth=1.0,
            zorder=-10,
        )
    )


def save_exports(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png")
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    plt.close(fig)


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[name], markeredgecolor="white", markeredgewidth=0.8, markersize=7.5, label=name)
        for name in ["IMERG", "ERA5", "EURADCLIM"]
    ]


def style_errorbar_axis(ax: plt.Axes, title: str) -> None:
    add_rounded_panel(ax)
    ax.set_title(title, loc="left", pad=12)
    ax.axvline(0.0, color=ZERO, linewidth=1.2, linestyle="--", zorder=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.9)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Delta MAE vs fixed local (mm)")


def plot_grouped_errorbars(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    style_errorbar_axis(ax, title)
    slices = df["slice_label"].drop_duplicates().tolist()
    products = ["IMERG", "ERA5", "EURADCLIM"]
    offsets = {"IMERG": -0.22, "ERA5": 0.0, "EURADCLIM": 0.22}
    y_base = np.arange(len(slices))[::-1].astype(float)

    for product in products:
        sub = df[df["eo_label"] == product].copy()
        y_values = []
        x_values = []
        xerr_low = []
        xerr_high = []
        for idx, slice_name in enumerate(slices):
            row = sub[sub["slice_label"] == slice_name].iloc[0]
            y = y_base[idx] + offsets[product]
            delta = float(row["delta_mae_eo_minus_local"])
            y_values.append(y)
            x_values.append(delta)
            xerr_low.append(delta - float(row["ci_low"]))
            xerr_high.append(float(row["ci_high"]) - delta)
        ax.errorbar(
            x_values,
            y_values,
            xerr=np.vstack([xerr_low, xerr_high]),
            fmt="o",
            color=COLORS[product],
            ecolor=COLORS[product],
            elinewidth=1.8,
            capsize=3.5,
            markersize=7.0,
            markerfacecolor=COLORS[product],
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )

    ax.set_yticks(y_base)
    ax.set_yticklabels(slices)
    lim = max(abs(float(df["ci_low"].min())), abs(float(df["ci_high"].max())))
    left = -0.03 if float(df["ci_low"].min()) < 0 else -0.01
    ax.set_xlim(left, lim * 1.10)
    ax.legend(
        handles=legend_handles(),
        loc="upper right",
        frameon=False,
        ncol=1,
        handletextpad=0.4,
        borderaxespad=0.3,
    )


def plot_best_eo_bars(ax: plt.Axes, df: pd.DataFrame, title: str, xlim: tuple[float, float]) -> None:
    add_rounded_panel(ax)
    ax.set_title(title, loc="left", pad=12)
    ax.axvline(0.0, color=ZERO, linewidth=1.1, linestyle="--", zorder=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.9)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(*xlim)

    y = np.arange(len(df))[::-1].astype(float)
    values = df["delta_mae_mm"].astype(float).to_numpy()
    colors = [COLORS[str(v)] for v in df["best_eo_label"]]
    ax.barh(y, values, color=colors, alpha=0.85, edgecolor="none", height=0.56, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(df["row_label"].tolist())

    span = xlim[1] - xlim[0]
    for yi, value in zip(y, values):
        if value >= 0:
            x_text = min(value + span * 0.02, xlim[1] - span * 0.08)
            ha = "left"
        else:
            x_text = max(value - span * 0.02, xlim[0] + span * 0.02)
            ha = "right"
        ax.text(
            x_text,
            yi,
            f"{value:+.3f}",
            va="center",
            ha=ha,
            fontsize=10,
            fontweight="bold",
            color=INK,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": BG, "edgecolor": "none", "alpha": 0.92},
            clip_on=False,
        )


def build_figure_1(panel_a: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    plot_grouped_errorbars(ax, panel_a, "Global and positive-rain penalties")
    save_exports(fig, "figure_1_global_positive_rain_v2")


def build_figure_2(panel_b: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    plot_grouped_errorbars(ax, panel_b, "Conflict-conditioned penalties")
    save_exports(fig, "figure_2_conflict_regimes_v2")


def build_figure_3(panel_c: pd.DataFrame) -> None:
    overall_conflict = panel_c[panel_c["row_group"].isin(["Overall", "Conflict"])].copy()
    intensity = panel_c[panel_c["row_group"] == "Intensity"].copy()

    fig = plt.figure(figsize=(8.6, 7.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.35], hspace=0.34, top=0.90)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0])

    plot_best_eo_bars(ax_top, overall_conflict, "Overall and conflict", (-0.08, 0.62))
    plot_best_eo_bars(ax_bottom, intensity, "Positive-rain intensity", (0.0, 12.0))

    ax_top.set_xlabel("")
    ax_bottom.set_xlabel("Delta MAE vs best local (mm)")

    fig.suptitle("Best EO by regime", x=0.12, y=0.985, ha="left", fontsize=16, fontweight="bold")
    fig.legend(
        handles=legend_handles(),
        loc="upper right",
        bbox_to_anchor=(0.985, 0.987),
        frameon=False,
        title="Best EO",
    )

    save_exports(fig, "figure_3_best_eo_by_regime_v2")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_a = pd.read_csv(PANEL_A_CSV)
    panel_b = pd.read_csv(PANEL_B_CSV)
    panel_c = pd.read_csv(PANEL_C_CSV)

    build_figure_1(panel_a)
    build_figure_2(panel_b)
    build_figure_3(panel_c)

    (OUT_DIR / "figure_manifest.json").write_text(
        json.dumps(
            {
                "source_dir": str(DATA_DIR),
                "outputs": {
                    "figure_1_global_positive_rain_v2": "Global and positive-rain delta MAE vs fixed local",
                    "figure_2_conflict_regimes_v2": "Conflict-conditioned delta MAE vs fixed local",
                    "figure_3_best_eo_by_regime_v2": "Best EO by regime as delta MAE vs best local, split into overall/conflict and intensity",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
