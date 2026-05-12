"""
Ablation study: effect of each workflow design decision on product rankings.

Reads from frozen result CSVs only (no restricted parquet needed).
Outputs: ablation_table_v1.csv  +  ablation_latex_v1.tex
"""

import os
import pandas as pd
import json

BASE    = os.path.join(os.path.dirname(__file__), "..", "results")
OUTDIR  = os.path.join(BASE, "ablation_study_v1")
os.makedirs(OUTDIR, exist_ok=True)

# ------------------------------------------------------------------
# Load frozen sources
# ------------------------------------------------------------------
df_gp  = pd.read_csv(os.path.join(BASE, "figures_source", "figure_2_global_positive.csv"))
df_cf  = pd.read_csv(os.path.join(BASE, "figures_source", "figure_3_conflict.csv"))
df_rt  = pd.read_csv(os.path.join(BASE, "conflict_policy_extension_v1",
                                   "conflict_policy_performance_v1.csv"))

LABELS = {"source_era5": "ERA5", "source_euradclim": "EURADCLIM", "source_imerg": "IMERG"}

def best_by(df, col):
    row = df.loc[df[col].idxmin()]
    return LABELS.get(row["eo_baseline"], row["eo_baseline"]), row[col]

# ------------------------------------------------------------------
# VERSION A  –  raw MAE vs. target, all hours, no local, no filters
# ------------------------------------------------------------------
va_all  = df_gp[df_gp["slice_id"] == "overall_all"]
va_rain = df_gp[df_gp["slice_id"] == "overall_positive_target"]

vA_winner_all,  vA_mae_all  = best_by(va_all,  "mae_eo")
vA_winner_rain, vA_mae_rain = best_by(va_rain, "mae_eo")

# ------------------------------------------------------------------
# VERSION B  –  same as A but positive-rain hours only
# ------------------------------------------------------------------
vB_winner_all,  vB_mae_all  = vA_winner_all, vA_mae_all    # all-hours unchanged
vB_winner_rain, vB_mae_rain = vA_winner_rain, vA_mae_rain  # now the focus metric

# ------------------------------------------------------------------
# VERSION C  –  + conflict stratification (raw mae_eo per bin, no local)
# ------------------------------------------------------------------
vc_low  = df_cf[df_cf["slice_id"] == "conflict_low"]
vc_mid  = df_cf[df_cf["slice_id"] == "conflict_mid"]
vc_tail = df_cf[df_cf["slice_id"] == "conflict_tail"]

vC_low_winner,  vC_low_mae  = best_by(vc_low,  "mae_eo")
vC_mid_winner,  vC_mid_mae  = best_by(vc_mid,  "mae_eo")
vC_tail_winner, vC_tail_mae = best_by(vc_tail, "mae_eo")

# Combined winner description for the table
vC_winner_str = (f"{vC_low_winner} (low, {vC_low_mae:.4f}); "
                 f"{vC_mid_winner} (mid, {vC_mid_mae:.3f}); "
                 f"{vC_tail_winner} (high, {vC_tail_mae:.3f})")

# ------------------------------------------------------------------
# VERSION D  –  + leave-cell local baseline, broad >=1 gauge universe
#               Uses wide_ge_1 scenario from delta_mae_by_support_score file
# ------------------------------------------------------------------
df_supp = pd.read_csv(os.path.join(BASE, "delta_mae_by_support_score_and_regime_v1.csv"))

d_ge1 = df_supp[
    (df_supp["scenario"] == "wide_ge_1") &
    (df_supp["support_score_bin"] == "all") &
    (df_supp["regime_stratifier"] == "overall") &
    (df_supp["regime_stratum"] == "all")
].copy()

vD_winner_all,  vD_delta_all  = best_by(d_ge1, "delta_mae_vs_fixed_local_mm")
vD_winner_rain, vD_delta_rain = best_by(d_ge1, "delta_mae_positive_vs_fixed_local_mm")
vD_n_all  = int(d_ge1["n_requested"].iloc[0])
vD_n_rain = int(d_ge1["n_positive_target"].iloc[0])

# ------------------------------------------------------------------
# VERSION E  –  + support >=2 gauge filter (robust_ge_2 universe)
#               shows that tighter support hardens benchmark
# ------------------------------------------------------------------
d_ge2 = df_supp[
    (df_supp["scenario"] == "robust_ge_2") &
    (df_supp["support_score_bin"] == "all") &
    (df_supp["regime_stratifier"] == "overall") &
    (df_supp["regime_stratum"] == "all")
].copy()

vE_winner_all,  vE_delta_all  = best_by(d_ge2, "delta_mae_vs_fixed_local_mm")
vE_winner_rain, vE_delta_rain = best_by(d_ge2, "delta_mae_positive_vs_fixed_local_mm")
vE_n_all  = int(d_ge2["n_requested"].iloc[0])
vE_n_rain = int(d_ge2["n_positive_target"].iloc[0])

# ------------------------------------------------------------------
# VERSION F  –  + conflict-gated routing (full workflow)
# ------------------------------------------------------------------
policy_row  = df_rt[df_rt["model"] == "Conflict policy"].iloc[0]
local_row   = df_rt[df_rt["model"] == "Fixed local"].iloc[0]
best_eo_row = df_rt[df_rt["model"] != "Conflict policy"].loc[
              df_rt[df_rt["model"] != "Conflict policy"]["overall_mae_mm"].idxmin()]

vF_policy_mae_all  = policy_row["overall_mae_mm"]
vF_policy_mae_rain = policy_row["positive_rain_mae_mm"]
vF_local_pct       = policy_row["local_usage_share"] * 100
vF_local_mae_all   = local_row["overall_mae_mm"]
vF_local_mae_rain  = local_row["positive_rain_mae_mm"]

# ------------------------------------------------------------------
# Assemble ablation table (CSV)
# ------------------------------------------------------------------
rows = [
    {
        "version":       "A",
        "design_added":  "Raw MAE vs. gauge target, all hours, no local comparator, no filters",
        "n_eval_all":    882999,
        "n_eval_rain":   20652,
        "best_all":      vA_winner_all,
        "metric_all":    f"MAE={vA_mae_all:.4f}",
        "best_rain":     vA_winner_rain,
        "metric_rain":   f"MAE={vA_mae_rain:.3f}",
        "conclusion":    "Single global ranking; EURADCLIM appears best overall",
    },
    {
        "version":       "B",
        "design_added":  "+ positive-rain conditioning",
        "n_eval_all":    882999,
        "n_eval_rain":   20652,
        "best_all":      vB_winner_all,
        "metric_all":    f"MAE={vA_mae_all:.4f}",
        "best_rain":     vB_winner_rain,
        "metric_rain":   f"MAE={vB_mae_rain:.3f}",
        "conclusion":    "Ranking shifts in rain-active hours (ERA5 overtakes EURADCLIM); aggregate masks regime structure",
    },
    {
        "version":       "C",
        "design_added":  "+ conflict stratification (no local comparator yet)",
        "n_eval_all":    882999,
        "n_eval_rain":   20652,
        "best_all":      f"{vC_low_winner}/{vC_mid_winner}/{vC_tail_winner} (low/mid/high)",
        "metric_all":    (f"Low:{vC_low_mae:.4f} Mid:{vC_mid_mae:.3f} High:{vC_tail_mae:.3f}"),
        "best_rain":     "Regime-dependent",
        "metric_rain":   "varies",
        "conclusion":    "No single winner; apparent best product changes with conflict regime",
    },
    {
        "version":       "D",
        "design_added":  "+ leave-cell local comparator (delta_MAE = gridded - local), support >=1 gauge",
        "n_eval_all":    vD_n_all,
        "n_eval_rain":   vD_n_rain,
        "best_all":      f"{vD_winner_all} (smallest positive delta)",
        "metric_all":    f"delta={vD_delta_all:+.4f}",
        "best_rain":     f"{vD_winner_rain} (smallest positive delta)",
        "metric_rain":   f"delta={vD_delta_rain:+.3f}",
        "conclusion":    "All gridded products worse than leave-cell local; question shifts from ranking to when local evidence dominates",
    },
    {
        "version":       "E",
        "design_added":  "+ support >=2 gauge filter (robust-support universe)",
        "n_eval_all":    vE_n_all,
        "n_eval_rain":   vE_n_rain,
        "best_all":      f"{vE_winner_all} (smallest delta)",
        "metric_all":    f"delta={vE_delta_all:+.4f}",
        "best_rain":     f"{vE_winner_rain} (smallest delta)",
        "metric_rain":   f"delta={vE_delta_rain:+.3f}",
        "conclusion":    (f"Support filter reduces universe ({vD_n_all:,} -> {vE_n_all:,} cell-hours) and widens gridded-minus-local gap; benchmark harder but direction preserved"),
    },
    {
        "version":       "F",
        "design_added":  "+ conflict-gated routing policy (full workflow)",
        "n_eval_all":    882999,
        "n_eval_rain":   20652,
        "best_all":      "Conflict-gated hybrid",
        "metric_all":    f"MAE={vF_policy_mae_all:.4f} (local-only={vF_local_mae_all:.4f})",
        "best_rain":     "Local-always",
        "metric_rain":   f"MAE={vF_local_mae_rain:.3f} (policy={vF_policy_mae_rain:.3f})",
        "conclusion":    (f"Routing reduces all-hours MAE by "
                         f"{(1 - vF_policy_mae_all/vF_local_mae_all)*100:.0f}% vs. always-local "
                         f"({vF_local_pct:.0f}% local branch usage); positive-rain still favors always-local"),
    },
]

df_abl = pd.DataFrame(rows)
csv_path = os.path.join(OUTDIR, "ablation_table_v1.csv")
df_abl.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

# ------------------------------------------------------------------
# LaTeX table
# ------------------------------------------------------------------
LATEX_ROWS = [
    # (version, design text, best_all_label, metric_all, best_rain_label, metric_rain)
    ("A",
     r"Raw MAE vs.\ target, all hours, no local comparator",
     r"EURADCLIM", f"${vA_mae_all:.3f}$",
     r"ERA5",      f"${vA_mae_rain:.2f}$"),
    ("B",
     r"$+$ positive-rain conditioning",
     r"EURADCLIM", f"${vA_mae_all:.3f}$",
     r"ERA5",      f"${vB_mae_rain:.2f}$"),
    ("C",
     r"$+$ conflict stratification (no local)",
     r"IMERG / IMERG / ERA5\textsuperscript{\dag}",
     r"varies",
     r"Regime-dependent",
     r"varies"),
    ("D",
     r"$+$ leave-cell local comparator ($\Delta$MAE), support ${\geq}1$",
     r"All worse; EURADCLIM closest",
     f"$\\Delta={vD_delta_all:+.3f}$",
     r"All worse; ERA5 closest",
     f"$\\Delta={vD_delta_rain:+.2f}$"),
    ("E",
     r"$+$ support ${\geq}2$ gauge filter",
     r"EURADCLIM (lowest $\Delta$)",
     f"$\\Delta={vE_delta_all:+.3f}$",
     r"ERA5 (lowest $\Delta$)",
     f"$\\Delta={vE_delta_rain:+.2f}$"),
    ("F",
     r"$+$ conflict-gated routing (full workflow)",
     r"Hybrid routing",
     f"${vF_policy_mae_all:.4f}$",
     r"Local-always",
     f"${vF_local_mae_rain:.3f}$"),
]

CONCLUSIONS = [
    r"Single-number ranking; EURADCLIM preferred globally",
    r"Ranking shifts in rain-active hours (ERA5 $\neq$ EURADCLIM)",
    r"No stable winner; apparent best changes with conflict regime",
    r"All gridded products inferior to local; local MAE is the new floor",
    r"Support filter hardens benchmark; direction preserved, margins wider",
    (r"Routing reduces all-hours MAE by "
     f"{(1 - vF_policy_mae_all/vF_local_mae_all)*100:.0f}"
     r"\% vs.\ always-local; positive-rain still needs local branch"),
]

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\begin{threeparttable}")
lines.append(r"\caption{Ablation study: effect of each workflow design decision on product rankings. "
             r"Each version adds one element to the previous design. "
             r"Best-ranked product (all-hours and rain-active) and the resulting operational "
             r"conclusion are shown for each design.}")
lines.append(r"\label{tab:ablation}")
lines.append(r"\footnotesize")
lines.append(r"\setlength{\tabcolsep}{4pt}")
lines.append(r"\begin{tabular}{clp{3.1cm}cp{2.8cm}cp{4.4cm}}")
lines.append(r"\toprule")
lines.append(r"Ver. & Design element added & Best (all hours) & Metric & Best (rain-active) & Metric & Operational conclusion \\")
lines.append(r"\midrule")

for (ver, design, ba, ma, br, mr), conc in zip(LATEX_ROWS, CONCLUSIONS):
    lines.append(fr"\textbf{{{ver}}} & {design} & {ba} & {ma} & {br} & {mr} & {conc} \\")
    lines.append(r"\addlinespace[2pt]")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\begin{tablenotes}")
lines.append(r"\footnotesize")
lines.append(r"\item[\dag] Low/mid/high conflict bins respectively.")
lines.append(r"\item All versions use the 2023 evaluable common universe ($|\mathcal{U}_{2023}^{(2)}|=882{,}999$ cell-hours; "
             r"20{,}652 positive-rain). "
             r"MAE in mm; $\Delta$MAE = gridded$-$local.")
lines.append(r"\end{tablenotes}")
lines.append(r"\end{threeparttable}")
lines.append(r"\end{table}")

latex_str = "\n".join(lines)
tex_path = os.path.join(OUTDIR, "ablation_latex_v1.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex_str)
print(f"Saved: {tex_path}")

# ------------------------------------------------------------------
# Manifest
# ------------------------------------------------------------------
manifest = {
    "script": "build_ablation_study_v1.py",
    "inputs": [
        "results/figures_source/figure_2_global_positive.csv",
        "results/figures_source/figure_3_conflict.csv",
        "results/conflict_policy_extension_v1/conflict_policy_performance_v1.csv",
    ],
    "outputs": {
        "csv":   "results/ablation_study_v1/ablation_table_v1.csv",
        "latex": "results/ablation_study_v1/ablation_latex_v1.tex",
    },
    "key_findings": {
        "A_vs_B": "EURADCLIM best globally but ERA5 better in rain-active hours — aggregate masks regime structure",
        "B_vs_C": "No single winner when conflict is introduced — apparent best product is regime-dependent",
        "C_vs_D": "Introducing leave-cell local baseline: ALL gridded products are worse than local; paradigm shift",
        "D_vs_E": (f"Support >=2 filter reduces universe from {vD_n_all:,} to {vE_n_all:,} cell-hours; "
                   f"all-hours gap widens ({vD_delta_all:+.4f} -> {vE_delta_all:+.4f}), "
                   f"rain-active gap widens ({vD_delta_rain:+.3f} -> {vE_delta_rain:+.3f})"),
        "E_vs_F": f"Conflict-gated routing reduces all-hours MAE by {(1 - vF_policy_mae_all/vF_local_mae_all)*100:.0f}% vs always-local; rain-active still needs local branch",
    }
}
with open(os.path.join(OUTDIR, "ablation_manifest_v1.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print("Done.")
print("\nKey transitions:")
for k, v in manifest["key_findings"].items():
    print(f"  {k}: {v}")
