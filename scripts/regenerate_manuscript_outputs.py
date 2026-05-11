#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
BOOTSTRAP = RESULTS / "bootstrap"
FIGURES_SOURCE = RESULTS / "figures_source"
MANIFESTS = RESULTS / "manifests"
DATA_FROZEN = ROOT / "data" / "frozen"


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    expected = "\n".join(f"- {path}" for path in paths)
    raise FileNotFoundError(f"None of the expected source files exists:\n{expected}")


def copy_csv(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)


def write_bootstrap_tables() -> None:
    source = first_existing(
        RESULTS / "bootstrap_delta_mae_v1" / "bootstrap_summary.csv",
        BOOTSTRAP / "bootstrap_delta_mae_summary.csv",
    )
    df = pd.read_csv(source)
    copy_csv(source, BOOTSTRAP / "bootstrap_delta_mae_summary.csv")

    fixed = df[df["comparator_kind"] == "fixed_local"].copy()
    main = fixed[fixed["slice_id"].isin(["overall_all", "overall_positive_target"])].copy()
    conflict = fixed[fixed["slice_id"].isin(["conflict_low", "conflict_mid", "conflict_tail"])].copy()
    main.to_csv(TABLES / "main_delta_mae_table.csv", index=False)
    main.to_csv(TABLES / "table_3_main_delta_mae.csv", index=False)
    conflict.to_csv(TABLES / "conflict_delta_mae_table.csv", index=False)
    conflict.to_csv(TABLES / "table_4_conflict_delta_mae.csv", index=False)


def write_table_outputs() -> None:
    copy_csv(
        first_existing(
            RESULTS / "conflict_policy_extension_v1" / "conflict_policy_performance_v1.csv",
            TABLES / "source_routing_summary.csv",
        ),
        TABLES / "source_routing_summary.csv",
    )
    copy_csv(TABLES / "source_routing_summary.csv", TABLES / "routing_summary.csv")

    copy_csv(
        first_existing(
            RESULTS / "best_eo_vs_best_local_by_regime_v1.csv",
            TABLES / "best_eo_vs_best_local_by_regime.csv",
        ),
        TABLES / "best_eo_vs_best_local_by_regime.csv",
    )

    day_block = first_existing(
        RESULTS / "bootstrap_block_sensitivity_v1" / "block_bootstrap_sensitivity_wide_v1.csv",
        BOOTSTRAP / "day_block_sensitivity.csv",
    )
    copy_csv(day_block, BOOTSTRAP / "day_block_sensitivity.csv")


def write_figure_sources() -> None:
    mappings = {
        "figure_2_global_positive_source.csv": (
            RESULTS / "main_figure_v1" / "panel_a_global_and_positive_rain.csv",
            FIGURES_SOURCE / "figure_2_global_positive.csv",
        ),
        "figure_3_conflict_source.csv": (
            RESULTS / "main_figure_v1" / "panel_b_conflict_bootstrap.csv",
            FIGURES_SOURCE / "figure_3_conflict.csv",
        ),
        "figure_s1_best_eo_by_regime.csv": (
            RESULTS / "main_figure_v1" / "panel_c_best_eo_by_regime.csv",
            FIGURES_SOURCE / "figure_s1_best_eo_by_regime.csv",
        ),
    }
    for public_name, candidates in mappings.items():
        source = first_existing(*candidates)
        copy_csv(source, FIGURES_SOURCE / public_name)
        if public_name.endswith("_source.csv"):
            copy_csv(source, FIGURES_SOURCE / public_name.replace("_source.csv", ".csv"))


def write_frozen_manifest() -> None:
    rows = [
        {
            "artifact": "canonical_cell_hour_benchmark_table",
            "public_path": "",
            "availability": "available from corresponding author subject to source-data conditions",
            "role": "canonical cell-hour table with gridded predictors, AVAMET target aggregate, support metadata, and disagreement fields",
        },
        {
            "artifact": "main_delta_mae_table",
            "public_path": "results/tables/main_delta_mae_table.csv",
            "availability": "public frozen result table",
            "role": "main global and positive-rain delta MAE contrasts",
        },
        {
            "artifact": "conflict_delta_mae_table",
            "public_path": "results/tables/conflict_delta_mae_table.csv",
            "availability": "public frozen result table",
            "role": "conflict-conditioned delta MAE contrasts",
        },
        {
            "artifact": "bootstrap_delta_mae_summary",
            "public_path": "results/bootstrap/bootstrap_delta_mae_summary.csv",
            "availability": "public bootstrap summary",
            "role": "hour-block bootstrap intervals for EO-local contrasts",
        },
        {
            "artifact": "figure_source_tables",
            "public_path": "results/figures_source/",
            "availability": "public figure-source tables",
            "role": "CSV sources used to regenerate manuscript figures",
        },
    ]
    for target in [MANIFESTS / "frozen_inputs_manifest.csv", DATA_FROZEN / "frozen_inputs_manifest.csv"]:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_output_manifest() -> None:
    outputs = sorted(
        path.relative_to(ROOT).as_posix()
        for directory in [TABLES, BOOTSTRAP, FIGURES_SOURCE, MANIFESTS]
        for path in directory.glob("*.csv")
    )
    manifest = {
        "description": "Manuscript-level frozen outputs regenerated from repository artefacts.",
        "outputs": outputs,
        "restricted_inputs_not_public": [
            "canonical cell-hour benchmark table",
            "AVAMET-derived local-baseline reconstruction layers",
        ],
    }
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    (MANIFESTS / "manuscript_outputs_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    for directory in [TABLES, BOOTSTRAP, FIGURES_SOURCE, MANIFESTS, DATA_FROZEN]:
        directory.mkdir(parents=True, exist_ok=True)
    write_bootstrap_tables()
    write_table_outputs()
    write_figure_sources()
    write_frozen_manifest()
    write_output_manifest()


if __name__ == "__main__":
    main()
