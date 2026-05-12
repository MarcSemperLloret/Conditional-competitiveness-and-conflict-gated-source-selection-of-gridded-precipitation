# An auditable geospatial benchmarking workflow for gridded precipitation products under dense-gauge co-availability

This repository accompanies the manuscript:

"An auditable geospatial benchmarking workflow for gridded precipitation products under dense-gauge co-availability"

The repository provides an installable Python package (`precipbench`), versioned analysis scripts, frozen manuscript-level outputs, and figure-source tables needed to audit and regenerate the manuscript-level analyses.

## precipbench package

The core computational functions are packaged as `precipbench` and can be installed directly from this repository:

```bash
pip install git+https://github.com/MarcSemperLloret/Conditional-competitiveness-and-conflict-gated-source-selection-of-gridded-precipitation.git
```

Or in editable mode after cloning:

```bash
git clone https://github.com/MarcSemperLloret/Conditional-competitiveness-and-conflict-gated-source-selection-of-gridded-precipitation.git
cd Conditional-competitiveness-and-conflict-gated-source-selection-of-gridded-precipitation
pip install -e .
```

The package exposes three modules:

| Module | Key functions |
|---|---|
| `precipbench.interpolation` | `haversine_km`, `build_neighbor_table`, `leavecell_idw` |
| `precipbench.metrics` | `conflict_index`, `compute_mae`, `compute_delta_mae` |
| `precipbench.universe` | `filter_common_universe`, `bin_conflict`, `bin_support` |

A minimal usage example:

```python
import numpy as np
import precipbench as pb

# Build leave-cell neighbor lookup (once per study domain)
neighbor_idx, neighbor_dist = pb.build_neighbor_table(
    cell_lats, cell_lons,
    station_lats, station_lons,
    station_cell_ids=station_cell_ids,
    cell_ids=cell_ids,
    k=64,
)

# Predict leave-cell IDW for all cell-hours
local_pred = pb.leavecell_idw(obs_matrix, neighbor_idx, neighbor_dist)

# Compute ΔΔMAEfor a gridded product
result = pb.compute_delta_mae(y_true, gridded_pred, local_pred, rain_only=False)
print(result)  # {'n': ..., 'n_rain': ..., 'mae_gridded': ..., 'delta_mae': ...}

# Conflict index across products
kappa = pb.conflict_index({"IMERG": imerg, "ERA5": era5, "EURADCLIM": euradclim})
regime = pb.bin_conflict(kappa)  # array of 'low' / 'mid' / 'high'
```

Run the unit tests with:

```bash
pip install pytest
pytest tests/
```

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

The canonical cell-hour benchmark table itself is not a public artefact in this repository. The public manifest and data dictionary are in `data/frozen/`.

## Repository Layout

- `scripts/`: versioned analysis and regeneration scripts.
- `results/`: frozen manuscript-level result tables, bootstrap summaries, figure-source tables, and manifests.
- `data/`: public data notes and restricted-input placeholders.
- `protocol/`: current benchmark and reproducibility protocols.
- `archive/`: older protocol notes retained for provenance only.
