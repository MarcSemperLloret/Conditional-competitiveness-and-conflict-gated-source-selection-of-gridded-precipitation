# Density Thresholds - Paper 11 Replication Repository

This repository contains the necessary scripts and protocol documents to reproduce the experiments for **Paper 11**, targeted for *Remote Sensing Applications: Society and Environment*.

## Research Question

This study investigates the conditions under which gridded Earth Observation (EO) precipitation products maintain operational utility compared to local gauge-assisted estimators, particularly when in-situ support is heterogeneous.

## Repository Structure

- `src/`: Python source code containing all the scripts to generate the tables, bootstrap analyses, metrics, and figure plotting packages.
- `protocol/`: Experimental protocols detailing the primary metrics, ranking methodologies, regimes (intensity, conflict, support), and rules for comparison.
- `data/`: Directory where the pre-processed input `.parquet` files should be placed (data not included, subject to AVAMET licensing).
- `results/`: Directory created during execution to store outputs such as CSV tables, ranking documents, and plotted figures.

## Data Access

The original AVAMET raw dataset and necessary pre-processed components are licensed and not redistributed directly in this repository. Ensure that the required data subsets are correctly placed in the `data/` directory as specified in the protocol documents before running the scripts. Access to raw data may be granted by AVAMET upon reasonable request.

## Setup Instructions

1. **Environment Setup**: Python 3.10+ is recommended. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Execution**: Scripts located in `src/` are typically run sequentially based on the desired target (table generation, metric derivation, or plotting). Please consult `protocol/support_utility_protocol_v1.md` for specific guidance on the exact pipelines.


