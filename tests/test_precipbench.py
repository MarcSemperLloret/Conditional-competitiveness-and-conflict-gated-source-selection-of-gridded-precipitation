"""Smoke tests for precipbench public API."""
import numpy as np
import pytest

import precipbench as pb


def test_haversine_known_distance():
    # Barcelona to Valencia ≈ 304 km
    d = pb.haversine_km(
        np.array([41.39]),
        np.array([2.15]),
        np.array([39.47]),
        np.array([-0.38]),
    )
    assert abs(d[0, 0] - 304) < 10


def test_conflict_index_three_products():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 2.0, 2.0])
    c = np.array([3.0, 2.0, 1.0])
    kappa = pb.conflict_index({"A": a, "B": b, "C": c})
    # row 0: |1-2|+|1-3|+|2-3| = 1+2+1 = 4, /3 ≈ 1.333
    assert abs(kappa[0] - 4 / 3) < 1e-4
    # row 1: all equal → 0
    assert kappa[1] == pytest.approx(0.0)


def test_compute_delta_mae_positive():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 5, 1000)
    gridded = y + rng.normal(0, 1, 1000)
    local = y + rng.normal(0, 0.5, 1000)
    result = pb.compute_delta_mae(y, gridded, local)
    assert result["delta_mae"] > 0   # gridded worse than local


def test_compute_delta_mae_rain_only():
    y = np.array([0.0, 0.0, 1.0, 2.0, 0.0])
    g = np.array([0.1, 0.1, 1.5, 2.5, 0.1])
    lo = np.array([0.0, 0.0, 1.1, 2.1, 0.0])
    result_all = pb.compute_delta_mae(y, g, lo, rain_only=False)
    result_rain = pb.compute_delta_mae(y, g, lo, rain_only=True)
    assert result_all["n"] == 5
    assert result_rain["n_rain"] == 2


def test_leavecell_idw_single_cell():
    # Two stations, one cell, single hour
    obs = np.array([[1.0, 2.0]], dtype=np.float32)  # (1 hour, 2 stations)
    idx = np.array([[0, 1]], dtype=np.int32)         # (1 cell, k=2)
    dist = np.array([[10.0, 20.0]], dtype=np.float32)
    pred = pb.leavecell_idw(obs, idx, dist, power=2.0)
    assert pred.shape == (1, 1)
    # w0=1/100, w1=1/400 → weighted mean closer to station 0
    assert pred[0, 0] == pytest.approx(
        (1.0 / 100 * 1.0 + 1.0 / 400 * 2.0) / (1.0 / 100 + 1.0 / 400),
        abs=1e-4,
    )


def test_bin_conflict_labels():
    kappa = np.linspace(0, 10, 1000)
    labels = pb.bin_conflict(kappa, q_low=0.75, q_high=0.95)
    assert set(labels) <= {"low", "mid", "high"}
    assert (labels == "low").sum() > (labels == "high").sum()


def test_bin_support_tiers():
    scores = np.array([0.1, 0.34, 0.5, 0.67, 0.9])
    tiers = pb.bin_support(scores)
    assert tiers[0] == "single_station_like"
    assert tiers[2] == "two_station_like"
    assert tiers[4] == "three_plus_like"
