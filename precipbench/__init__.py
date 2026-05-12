"""precipbench — auditable geospatial benchmarking for gridded precipitation.

Public API
----------
interpolation
    haversine_km          great-circle distance matrix
    build_neighbor_table  leakage-controlled k-NN lookup table
    leavecell_idw         leave-cell IDW prediction

metrics
    conflict_index        mean pairwise absolute product difference (κ)
    compute_mae           mean absolute error over finite pairs
    compute_delta_mae     gridded-minus-local ΔΔMAEwith rain-active option

universe
    filter_common_universe  evaluable cell-hour subset
    bin_conflict            low/mid/high conflict labels from κ
    bin_support             support-score tier labels
"""

from precipbench.interpolation import (
    build_neighbor_table,
    haversine_km,
    leavecell_idw,
)
from precipbench.metrics import (
    compute_delta_mae,
    compute_mae,
    conflict_index,
)
from precipbench.universe import (
    bin_conflict,
    bin_support,
    filter_common_universe,
)

__version__ = "1.0.0"
__all__ = [
    "haversine_km",
    "build_neighbor_table",
    "leavecell_idw",
    "conflict_index",
    "compute_mae",
    "compute_delta_mae",
    "filter_common_universe",
    "bin_conflict",
    "bin_support",
]
