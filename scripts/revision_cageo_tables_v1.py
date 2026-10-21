#!/usr/bin/env python
"""Emit LaTeX table bodies for the CAGEO revision from results/revision_cageo_v1 CSVs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from revision_cageo_analyses_v1 import EO_SOURCES, LOCAL_COL, SOURCES, bootstrap_delta, conflict_slices, robust_frame
from revision_cageo_cache_v1 import load_year


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "revision_cageo_v1"
OUT = RES / "tex"
ORDER = ["ERA5", "EURADCLIM", "IMERG"]


def s(x: float, d: int = 3) -> str:
    return f"{x:+.{d}f}"


def ci(r: pd.Series, d: int = 4) -> str:
    return f"[{r['ci_low']:+.{d}f}, {r['ci_high']:+.{d}f}]"


def main_tables() -> dict[str, str]:
    out = {}
    b = pd.read_csv(RES / "bootstrap_block_sensitivity_2023.csv").set_index(["slice", "source", "block"])

    # Table 2: global and positive rain with hour and wet-spell CIs
    lines = []
    for eo in ORDER:
        g_h, g_e = b.loc[("all hours", eo, "hour")], b.loc[("all hours", eo, "wet-spell event")]
        p_h, p_e = b.loc[("positive rain", eo, "hour")], b.loc[("positive rain", eo, "wet-spell event")]
        lines.append(f"{eo} & {s(g_h['delta'],4)} & {ci(g_h)} & {ci(g_e)} & {s(p_h['delta'],4)} & {ci(p_h)} & {ci(p_e)} \\\\")
    out["table_main"] = "\n".join(lines)

    # Table 3: conflict with hour and wet-spell CIs
    lines = []
    for reg, label in [("low conflict", "Low conflict"), ("mid conflict", "Mid conflict"), ("high conflict", "High conflict")]:
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textbf{{{label}}}}} \\\\")
        sub = b.xs((reg, "hour"), level=("slice", "block")).sort_values("delta")
        for eo, r in sub.iterrows():
            e = b.loc[(reg, eo, "wet-spell event")]
            lines.append(f"{eo} & {s(r['delta'],4)} & {ci(r)} & {ci(e)} \\\\")
        lines.append("\\addlinespace[3pt]")
    out["table_conflict"] = "\n".join(lines[:-1])

    # Multi-metric table
    m = pd.read_csv(RES / "metrics_multi_2023.csv")
    lines = []
    for slice_name in ["all hours", "positive rain"]:
        lines.append(f"\\multicolumn{{10}}{{l}}{{\\textbf{{{slice_name.capitalize()}}}}} \\\\")
        for _, r in m[m["slice"] == slice_name].iterrows():
            src = "Local IDW-64" if r["source"].startswith("Local") else r["source"]
            if slice_name == "all hours":
                cat = " & ".join(f"{r[f'csi_{t}mm']:.3f}" for t in ["0.1", "1", "5", "10"])
            else:
                cat = "\\multicolumn{4}{c}{---}"
            lines.append(f"{src} & {r['mae_mm']:.3f} & {r['rmse_mm']:.3f} & {r['bias_mm']:+.3f} & {r['r']:.2f} & {r['kge']:.2f} & {cat} \\\\")
        lines.append("\\addlinespace[3pt]")
    out["table_metrics"] = "\n".join(lines[:-1])

    # POD/FAR supplementary
    lines = []
    for _, r in m[m["slice"] == "all hours"].iterrows():
        src = "Local IDW-64" if r["source"].startswith("Local") else r["source"]
        cells = " & ".join(f"{r[f'pod_{t}mm']:.2f}/{r[f'far_{t}mm']:.2f}" for t in ["0.1", "1", "5", "10"])
        lines.append(f"{src} & {cells} \\\\")
    out["table_podfar"] = "\n".join(lines)
    return out


def robustness_table() -> str:
    rows = []
    rs = pd.read_csv(RES / "resampling_sensitivity_2023.csv").set_index(["config", "slice", "source"])
    kr = pd.read_csv(RES / "kriging_comparator_sensitivity_2023.csv")
    kr = kr[kr["block"] == "hour"].set_index(["local", "slice", "source"])
    cn = pd.read_csv(RES / "conflict_normalized_2023.csv")
    cn = cn[cn["subset"] == "all hours"].set_index(["index", "stratum", "source"])
    b = pd.read_csv(RES / "bootstrap_block_sensitivity_2023.csv").set_index(["slice", "source", "block"])

    def row(label, pos, low, high, note=""):
        rows.append(f"{label} & " + " & ".join(s(pos[e]) for e in ORDER) + f" & {low} & {high} \\\\")

    def rng(d):
        v = [d[e] for e in ORDER]
        return f"{min(v):+.3f} to {max(v):+.3f}" if abs(max(v) - min(v)) >= 0.0005 else f"{v[0]:+.3f}"

    base_cfg = "manuscript (ERA5 bilinear, EUR footprint mean)"
    for label, cfg in [("Manuscript design", base_cfg), ("ERA5 nearest neighbour", "ERA5 nearest"),
                       ("ERA5 first-order conservative", "ERA5 conservative"), ("EURADCLIM nearest pixel", "EURADCLIM nearest")]:
        pos = {e: rs.loc[(cfg, "positive rain", e), "delta"] for e in ORDER}
        low = {e: rs.loc[(cfg, "low conflict", e), "delta"] for e in ORDER}
        high = {e: rs.loc[(cfg, "high conflict", e), "delta"] for e in ORDER}
        row(label, pos, rng(low), rng(high))
    for label, lname in [("Leave-cell ordinary kriging", "OK-64"), ("Leave-cell KED (elevation drift)", "KED-64 (elevation)")]:
        pos = {e: kr.loc[(lname, "positive rain", e), "delta"] for e in ORDER}
        low = {e: kr.loc[(lname, "low conflict", e), "delta"] for e in ORDER}
        high = {e: kr.loc[(lname, "high conflict", e), "delta"] for e in ORDER}
        row(label, pos, rng(low), rng(high))
    # mean target
    nf = pd.read_csv(RES / "representativeness_noise_floor_2023.csv").set_index("slice")
    pos = {e: nf.loc["positive rain", f"{e}_delta_mae_mean_target"] for e in ORDER}
    rows.append("Within-cell mean target & " + " & ".join(s(pos[e]) for e in ORDER) + " & -- & -- \\\\")
    # relative conflict
    low = {e: cn.loc[("relative kappa", "low", e), "delta"] for e in ORDER}
    high = {e: cn.loc[("relative kappa", "high", e), "delta"] for e in ORDER}
    pos = {e: b.loc[("positive rain", e, "hour"), "delta"] for e in ORDER}
    rows.append("Relative conflict index $\\kappa^{\\mathrm{rel}}$ & \\multicolumn{3}{c}{(unchanged)} & " + f"{rng(low)} & {rng(high)} \\\\")
    return "\n".join(rows)


def years_table() -> str:
    py = pd.read_csv(RES / "per_year_extremes.csv")
    rt = pd.read_csv(RES / "routing_by_year_extremes.csv")
    lines = []
    for year in sorted(py["year"].unique()):
        sub = py[(py["year"] == year) & (py["stratum"] == "positive rain")].set_index("source")
        ext = py[(py["year"] == year) & (py["stratum"] == "target > 20 mm")].set_index("source")
        wet = sub["wet_share"].iloc[0] if "wet_share" in sub else py[py["year"] == year]["wet_share_year"].iloc[0]
        wet = py[py["year"] == year]["wet_share_year"].iloc[0]
        cells = " & ".join(s(sub.loc[e, "delta"], 2) for e in ORDER)
        best = ext["delta"].idxmin()
        bx = ext.loc[best]
        ext_cell = f"{best[:4] if best != 'EURADCLIM' else 'EUR.'} {bx['delta']:+.1f}"
        r = rt[(rt["test"] == year)]
        if r.empty:
            rcells = "\\multicolumn{2}{c}{(training only)}"
        else:
            a = r[r["slice"] == "all hours"].iloc[0]
            p = r[r["slice"] == "positive rain"].iloc[0]
            rcells = f"{100 * (a['mae_policy'] / a['mae_local'] - 1):+.0f}\\% & {p['mae_policy'] - p['mae_local']:+.3f}"
        lines.append(f"{year} & {100 * wet:.1f} & {cells} & {ext_cell} & {rcells} \\\\")
    return "\n".join(lines)


def threshold_table() -> str:
    t = pd.read_csv(RES / "conflict_threshold_sensitivity_2023.csv")
    lines = []
    for cut, g in t.groupby("cutoffs", sort=False):
        parts = []
        for st in ["low", "mid", "high"]:
            d = g[g["stratum"] == st].set_index("source")
            parts.append(f"{100 * d['wet_share'].iloc[0]:.2f}")
            parts.append(" / ".join(f"{d.loc[e, 'delta']:+.3f}" for e in ORDER))
        lines.append(f"{cut.replace('p', 'P')} & " + " & ".join(parts) + " \\\\")
    return "\n".join(lines)


def intensity_table() -> str:
    c = pd.read_csv(RES / "conflict_by_intensity_2023.csv")
    c = c[c["index"] == "relative kappa"]
    lines = []
    for ib, g in c.groupby("intensity_bin", sort=False):
        cells = []
        for e in ORDER:
            d = g[g["source"] == e].set_index("stratum")
            cells.append(" / ".join(f"{d.loc[t, 'delta']:+.2f}" for t in ["T1 (low)", "T2", "T3 (high)"]))
        n = int(g["n"].iloc[0]) * 3
        lines.append(f"{ib} mm & {n} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def geometry_table() -> str:
    g = pd.read_csv(RES / "representativeness_geometry_2023.csv")
    lines = []
    for st, d in g.groupby("stratum", sort=False):
        d = d.set_index("source")
        lines.append(f"{st} & {int(d['n_pos'].iloc[0])} & {d['mae_local'].iloc[0]:.3f} & " +
                     " & ".join(f"{d.loc[e, 'delta']:+.3f}" for e in ORDER) + " \\\\")
    return "\n".join(lines)


def block_table() -> str:
    b = pd.read_csv(RES / "bootstrap_block_sensitivity_2023.csv")
    lines = []
    for (sl, eo), d in b.groupby(["slice", "source"], sort=False):
        d = d.set_index("block")
        cells = " & ".join(f"[{d.loc[k, 'ci_low']:+.4f}, {d.loc[k, 'ci_high']:+.4f}]" for k in ["hour", "day", "7-day", "wet-spell event"])
        lines.append(f"{sl} & {eo} & {d.loc['hour', 'delta']:+.4f} & {cells} \\\\")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = main_tables()
    blocks |= {"table_robustness": robustness_table(), "table_years": years_table(), "table_thresholds": threshold_table(),
               "table_intensity": intensity_table(), "table_geometry": geometry_table(), "table_blocks": block_table()}
    for k, v in blocks.items():
        (OUT / f"{k}.tex").write_text(v + "\n", encoding="utf-8")
        print(f"%% ===== {k}\n{v}\n")


if __name__ == "__main__":
    main()
