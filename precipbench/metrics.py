"""Core benchmark metrics: ΔΔMAEand conflict index."""
from __future__ import annotations

import numpy as np


def conflict_index(
    products: dict[str, np.ndarray],
) -> np.ndarray:
    """Mean pairwise absolute difference across gridded products (κ).

    Parameters
    ----------
    products : mapping of product_name → (N,) precipitation array

    Returns
    -------
    (N,) float32 conflict index κ, NaN where any product is NaN

    Notes
    -----
    With three products A, B, C:
        κ = (|A−B| + |A−C| + |B−C|) / 3
    Generalises to any number of products via all unique pairs.
    """
    arrays = [np.asarray(v, dtype=np.float64) for v in products.values()]
    if len(arrays) < 2:
        raise ValueError("Need at least two products to compute conflict index.")

    n = arrays[0].size
    pair_sum = np.zeros(n, dtype=np.float64)
    pair_count = np.zeros(n, dtype=np.int32)

    names = list(products.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = arrays[i], arrays[j]
            both_finite = np.isfinite(a) & np.isfinite(b)
            pair_sum[both_finite] += np.abs(a[both_finite] - b[both_finite])
            pair_count[both_finite] += 1

    kappa = np.full(n, np.nan, dtype=np.float32)
    valid = pair_count > 0
    kappa[valid] = (pair_sum[valid] / pair_count[valid]).astype(np.float32)
    return kappa


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Mean absolute error over finite pairs."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return None
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask])))


def compute_delta_mae(
    y_true: np.ndarray,
    gridded: np.ndarray,
    local: np.ndarray,
    rain_only: bool = False,
) -> dict[str, float | int | None]:
    """Gridded-minus-local ΔΔMAEbenchmark metric.

    Parameters
    ----------
    y_true : (N,) gauge target (within-cell median)
    gridded : (N,) gridded product estimates
    local : (N,) leave-cell IDW local baseline
    rain_only : if True, restrict to positive-target hours (y_true > 0)

    Returns
    -------
    dict with keys:
        n            — number of evaluated cell-hours
        n_rain       — number of positive-target cell-hours
        mae_gridded  — gridded MAE in mm
        mae_local    — local baseline MAE in mm
        delta_mae    — gridded MAE − local MAE (positive = gridded worse)
    """
    y = np.asarray(y_true, dtype=np.float64)
    g = np.asarray(gridded, dtype=np.float64)
    lo = np.asarray(local, dtype=np.float64)

    base_mask = np.isfinite(y) & np.isfinite(g) & np.isfinite(lo)
    rain_mask = base_mask & (y > 0.0)

    mask = rain_mask if rain_only else base_mask

    n_all = int(base_mask.sum())
    n_rain = int(rain_mask.sum())

    if not mask.any():
        return {"n": n_all, "n_rain": n_rain,
                "mae_gridded": None, "mae_local": None, "delta_mae": None}

    mae_g = float(np.mean(np.abs(g[mask] - y[mask])))
    mae_l = float(np.mean(np.abs(lo[mask] - y[mask])))
    return {
        "n": n_all,
        "n_rain": n_rain,
        "mae_gridded": mae_g,
        "mae_local": mae_l,
        "delta_mae": mae_g - mae_l,
    }
