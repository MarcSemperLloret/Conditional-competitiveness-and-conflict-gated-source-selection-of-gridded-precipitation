#!/usr/bin/env python
"""Cache per-year prediction frames (canonical rows + leave-cell IDW baselines) for the CAGEO revision analyses."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from build_temporal_sensitivity_2022_v1 import build_year_frame


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "results" / "revision_cageo_v1" / "cache"
YEARS = [2019, 2020, 2021, 2022, 2023]


def cache_path(year: int) -> Path:
    return CACHE_DIR / f"prediction_frame_{year}.parquet"


def load_year(year: int) -> pd.DataFrame:
    path = cache_path(year)
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        build_year_frame(year).to_parquet(path, index=False)
    return pd.read_parquet(path)


def main() -> None:
    years = [int(y) for y in sys.argv[1:]] or YEARS
    for year in years:
        df = load_year(year)
        robust = df["avamet_n_active"] >= 2
        print(year, len(df), int(robust.sum()), float((df.loc[robust, "avamet_agg_mm"] > 0).mean()), flush=True)


if __name__ == "__main__":
    main()
