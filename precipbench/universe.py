"""Cell-hour universe filtering and stratification utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def filter_common_universe(
    df: pd.DataFrame,
    gridded_cols: list[str],
    target_col: str = "target_mm",
    min_gauges: int = 2,
    gauge_count_col: str = "n_gauges",
) -> pd.DataFrame:
    """Return the evaluable common universe from a cell-hour DataFrame.

    A cell-hour is included when:
      - target and all gridded columns are finite
      - gauge count >= min_gauges

    Parameters
    ----------
    df : cell-hour DataFrame
    gridded_cols : column names for gridded product estimates
    target_col : column name for the within-cell gauge target
    min_gauges : minimum active gauge count (default 2 = robust universe)
    gauge_count_col : column name for active gauge count

    Returns
    -------
    Filtered DataFrame (copy).
    """
    mask = pd.Series(True, index=df.index)
    mask &= df[target_col].notna() & np.isfinite(df[target_col])
    for col in gridded_cols:
        mask &= df[col].notna() & np.isfinite(df[col])
    mask &= df[gauge_count_col] >= min_gauges
    return df.loc[mask].copy()


def bin_conflict(
    kappa: np.ndarray,
    q_low: float = 0.75,
    q_high: float = 0.95,
) -> np.ndarray:
    """Assign conflict-regime labels to a κ array.

    Returns object array with values 'low', 'mid', 'high'.

    Parameters
    ----------
    kappa : (N,) conflict index values
    q_low : lower percentile boundary (default p75)
    q_high : upper percentile boundary (default p95)
    """
    finite = kappa[np.isfinite(kappa)]
    p_low = float(np.nanpercentile(finite, q_low * 100))
    p_high = float(np.nanpercentile(finite, q_high * 100))

    labels = np.full(kappa.size, "low", dtype=object)
    labels[(kappa > p_low) & (kappa <= p_high)] = "mid"
    labels[kappa > p_high] = "high"
    labels[~np.isfinite(kappa)] = "unknown"
    return labels


def bin_support(support_score: np.ndarray) -> np.ndarray:
    """Map support scores to three-tier labels.

    Thresholds follow the paper convention:
        score ≤ 0.34  → 'single_station_like'
        0.34 < score ≤ 0.67 → 'two_station_like'
        score > 0.67  → 'three_plus_like'
    """
    labels = np.full(support_score.size, "two_station_like", dtype=object)
    labels[support_score <= 0.34] = "single_station_like"
    labels[support_score > 0.67] = "three_plus_like"
    return labels
