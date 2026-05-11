# A reproducible geospatial workflow for benchmarking gridded precipitation products against dense gauge networks

This repository accompanies the manuscript:

"A reproducible geospatial workflow for benchmarking gridded precipitation products against dense gauge networks"

The repository provides the scripts, configuration files, frozen manuscript-level outputs, figure-source tables, and selected derived benchmark artefacts needed to audit and regenerate the manuscript-level analyses.

## What Can Be Reproduced

The public frozen artefacts allow regeneration of manuscript-level tables, bootstrap summaries, and figure-source files under the common cell-hour benchmark design. The main public outputs are written to:

- `results/tables/`
- `results/bootstrap/`
- `results/figures_source/`
- `results/manifests/`

## Restricted Inputs

Full reconstruction from raw AVAMET holdings is not possible from public files because the upstream station archive is restricted. The canonical cell-hour benchmark table and AVAMET-derived local-baseline reconstruction layers are available from the corresponding author upon reasonable request, subject to source-data conditions.

Scripts that rebuild the upstream benchmark layer expect those restricted files under `data/restricted/`, or a custom path supplied with `BENCHMARK_RESTRICTED_DATA_DIR`.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/regenerate_manuscript_outputs.py
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Expected Outputs

Running the regeneration script refreshes:

- `results/tables/main_delta_mae_table.csv`
- `results/tables/conflict_delta_mae_table.csv`
- `results/tables/source_routing_summary.csv`
- `results/bootstrap/bootstrap_delta_mae_summary.csv`
- `results/figures_source/figure_2_global_positive_source.csv`
- `results/figures_source/figure_3_conflict_source.csv`
- `results/figures_source/figure_s1_best_eo_by_regime.csv`
- `results/manifests/frozen_inputs_manifest.csv`

## Repository Layout

- `scripts/`: versioned analysis and regeneration scripts.
- `results/`: frozen manuscript-level result tables, bootstrap summaries, figure-source tables, and manifests.
- `data/`: public data notes and restricted-input placeholders.
- `protocol/`: current benchmark and reproducibility protocols.
- `archive/`: older protocol notes retained for provenance only.
