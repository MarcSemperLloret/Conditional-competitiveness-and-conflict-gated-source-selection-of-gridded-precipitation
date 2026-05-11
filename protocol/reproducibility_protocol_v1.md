# Reproducibility Protocol v1

The public repository supports manuscript-level regeneration from frozen outputs. It does not support end-to-end reconstruction from raw AVAMET holdings.

## Public Regeneration

Run:

```bash
pip install -r requirements.txt
python scripts/regenerate_manuscript_outputs.py
```

This refreshes the public result tables, bootstrap summaries, figure-source tables, and manifests under `results/`.

## Restricted Reconstruction

Scripts that rebuild local baselines and upstream benchmark assets require restricted AVAMET-derived files. Place those files in `data/restricted/` or set `BENCHMARK_RESTRICTED_DATA_DIR`.

The canonical cell-hour benchmark table and AVAMET-derived local-baseline layers are available from the corresponding author upon reasonable request, subject to source-data conditions.
