#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "main_figure_v1"
OUT_DIR = DATA_DIR

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
LOCAL = "#5a6570"
DELTA_FILL = "#f5f7fa"
DELTA_EDGE = "#e1e7ef"

COLORS = {
    "IMERG": IMERG,
    "ERA5": ERA5,
    "EURADCLIM": EURAD,
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Aptos", "Segoe UI", "DejaVu Sans"],
        "font.size": 10.5,
        "axes.titlesize": 13.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
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


def style_errorbar_axis(ax: plt.Axes, title: str) -> None:
    add_rounded_panel(ax)
    ax.set_title(title, loc="left", pad=10, fontsize=13.5)
    ax.axvline(0.0, color=ZERO, linewidth=1.2, linestyle="--", zorder=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.9)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Delta MAE vs local (mm)")


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
            markersize=6.5,
            markerfacecolor=COLORS[product],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=product,
            zorder=3,
        )

    ax.set_yticks(y_base)
    ax.set_yticklabels(slices)
    lim = max(
        abs(float(df["ci_low"].min())),
        abs(float(df["ci_high"].max())),
    )
    ax.set_xlim(-0.02 if lim < 0.05 else -0.03, lim * 1.08)
    ax.legend(loc="lower right", frameon=False, ncol=1, handletextpad=0.4)


def delta_fill_color(delta: float) -> str:
    if delta < 0:
        return "#d8f0df"
    if delta < 0.05:
        return "#eef4fb"
    if delta < 0.5:
        return "#fdeecf"
    return "#f9d8d0"


def plot_regime_table(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_rounded_panel(ax)
    ax.set_title("C. Best EO by regime", loc="left", pad=8, fontsize=13.5)

    columns = [
        ("Regime", 0.04, 0.25),
        ("Best EO", 0.31, 0.18),
        ("EO MAE", 0.52, 0.12),
        ("Local MAE", 0.67, 0.13),
        ("Delta", 0.83, 0.12),
    ]
    header_y = 0.89
    row_h = 0.075

    for title, x0, width in columns:
        ax.add_patch(Rectangle((x0, header_y), width, 0.07, facecolor="#eef3f8", edgecolor=GRID, linewidth=0.8))
        ax.text(x0 + 0.01, header_y + 0.035, title, va="center", ha="left", fontweight="bold", fontsize=10.5)

    current_group = None
    for idx, row in df.iterrows():
        y0 = header_y - (idx + 1) * row_h
        if row["row_group"] != current_group:
            current_group = row["row_group"]
            ax.text(0.04, y0 + row_h * 0.9, current_group.upper(), fontsize=8.5, color=MUTED, fontweight="bold")

        for _, x0, width in columns:
            ax.add_patch(Rectangle((x0, y0), width, row_h, facecolor="white", edgecolor=DELTA_EDGE, linewidth=0.6))

        delta = float(row["delta_mae_mm"])
        ax.add_patch(
            Rectangle(
                (0.83, y0),
                0.12,
                row_h,
                facecolor=delta_fill_color(delta),
                edgecolor=DELTA_EDGE,
                linewidth=0.6,
            )
        )
        ax.add_patch(
            Rectangle(
                (0.31, y0 + 0.01),
                0.16,
                row_h - 0.02,
                facecolor=COLORS[str(row["best_eo_label"])],
                edgecolor="none",
                alpha=0.16,
            )
        )

        ax.text(0.05, y0 + row_h * 0.5, str(row["row_label"]), va="center", ha="left", fontsize=10)
        ax.text(0.32, y0 + row_h * 0.5, str(row["best_eo_label"]), va="center", ha="left", fontsize=10, fontweight="bold", color=COLORS[str(row["best_eo_label"])])
        ax.text(0.53, y0 + row_h * 0.5, f"{float(row['best_eo_mae_mm']):.3f}", va="center", ha="left", fontsize=10)
        ax.text(0.68, y0 + row_h * 0.5, f"{float(row['best_local_mae_mm']):.3f}", va="center", ha="left", fontsize=10, color=LOCAL)
        ax.text(0.84, y0 + row_h * 0.5, f"{delta:+.3f}", va="center", ha="left", fontsize=10, fontweight="bold")

        ax.text(
        0.04,
        0.03,
        "Negative delta means the EO product beats the fixed or best local comparator on MAE.",
        fontsize=8.5,
        color=MUTED,
    )


def build_figure() -> None:
    panel_a = pd.read_csv(PANEL_A_CSV)
    panel_b = pd.read_csv(PANEL_B_CSV)
    panel_c = pd.read_csv(PANEL_C_CSV)

    fig = plt.figure(figsize=(10.2, 9.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.98, 1.22], hspace=0.24, wspace=0.25)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    plot_grouped_errorbars(ax_a, panel_a, "A. Global vs positive rain")
    plot_grouped_errorbars(ax_b, panel_b, "B. Conflict regimes")
    plot_regime_table(ax_c, panel_c)

    fig.suptitle(
        "Conditional utility of EO precipitation products against leave-cell local estimates",
        x=0.02,
        y=0.975,
        ha="left",
        fontsize=14.5,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.02,
        0.947,
        "Losses are modest globally but widen in positive rain and in higher-conflict regimes.",
        ha="left",
        fontsize=9.8,
        color=MUTED,
    )
    fig.subplots_adjust(top=0.83, bottom=0.06, left=0.07, right=0.985)

    save_exports(fig, "figure_1_regime_dependent_utility")


if __name__ == "__main__":
    build_figure()

