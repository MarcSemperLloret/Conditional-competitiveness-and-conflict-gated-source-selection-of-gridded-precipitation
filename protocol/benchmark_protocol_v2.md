# Benchmark Protocol v2

The benchmark evaluates IMERG, ERA5, and EURADCLIM against leave-cell AVAMET local estimates on a common hourly IMERG cell-hour universe.

## Evaluation Unit

Each row represents one IMERG cell and one UTC hour. The target is the within-cell AVAMET gauge aggregate, and predictors are harmonized gridded precipitation values plus derived support and disagreement metadata.

## Common Universe

The manuscript-level comparisons use cell-hours where the target aggregate and all three gridded products are finite. The main scenario requires at least two active AVAMET gauges in the target cell-hour; the one-gauge universe is retained as a sensitivity.

## Comparators

The inferential comparator is the fixed 64-neighbor leave-cell IDW local benchmark. The 8-neighbor and 64-neighbor within 15 km variants are used as sensitivity and descriptive best-local comparators.

## Regimes

Regime stratification uses target rainfall intensity, local support, and an observable inter-product disagreement score computed only from IMERG, ERA5, and EURADCLIM. The disagreement score is a routing and stratification marker, not an independent uncertainty estimate.
