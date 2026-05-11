# Data Availability

The public repository contains manuscript-level frozen result tables, bootstrap summaries, figure-source tables, and manifests.

The canonical cell-hour benchmark table and AVAMET-derived local-baseline reconstruction layers are not redistributed here because they depend on restricted upstream AVAMET holdings. They are available from the corresponding author upon reasonable request, subject to source-data conditions.

For local full reconstruction, place restricted inputs in `data/restricted/` or set `BENCHMARK_RESTRICTED_DATA_DIR` to the directory containing:

- `canonical_cell_hour_imerg_v1.parquet`
- `imerg_cell_static_v1.parquet`
- `avamet_station_inventory_cv_imerg.csv`
- `avamet_cv_hourly_2019_2023.parquet`
- `baseline_results_v3.csv`
