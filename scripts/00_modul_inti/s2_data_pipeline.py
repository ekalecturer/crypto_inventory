"""
Section 3.3 - Pengumpulan Data & Pra-pemrosesan (yfinance version).

CHANGE LOG vs the original Colab pipeline (which used CoinGecko):
  - Binance API is blocked in Indonesia (logbook 30/04/2026) -- not an option.
  - CoinGecko free tier: 365-day history cap. Paid (Analyst) tier: 2-year
    cap. Neither covers the 2022-2024 calibration window, which must
    include the FTX collapse (Nov 2022) per the proposal's own rationale
    for using bootstrap-MC over a parametric model.
  - yfinance (Yahoo Finance) has full BTC-USD/ETH-USD daily history back
    to ~2014/2017 respectively, no subscription needed. Trade-off: Yahoo's
    crypto volume figures are still a composite/estimate (no single
    exchange's real order-book volume), same caveat as CoinGecko's
    aggregated volume -- this is not a new limitation introduced by the
    switch, just carried over. Document it as such, don't claim exchange-
    specific ground truth.
  - The sandbox this pipeline runs in cannot reach Yahoo Finance directly
    (outbound network blocked at the proxy) -- see
    00_fetch_yfinance_LOCAL.py, which must be run on a machine with real
    internet access, with its output CSVs uploaded into data/raw/. This
    module reads those local CSVs; it does not call yfinance itself.

  - UNIT BUG (found on the real BTC/ETH CSVs, same class of bug the
    original CoinGecko pipeline already had to fix once): Yahoo Finance's
    `Volume` field for crypto tickers (BTC-USD, ETH-USD) is USD-denominated
    dollar volume, NOT a count of coins. Evidence: BTC_yfinance_raw.csv
    lists ~24.6 billion "volume" on 2022-01-01, and there are only ~19
    million BTC in existence -- that number cannot be base-asset units.
    Left unconverted, every downstream "inventory," "demand," and
    holding-cost figure would be off by a factor of the asset's price
    (thousands to tens of thousands x). Fix applied in load_raw_csv()
    below: volume_units = volume_usd / close, matching the same
    conversion the original pipeline applied to CoinGecko's total_volumes.

Preprocessing rules (proposal 3.3, unchanged):
    - gaps < 3 days  -> linear interpolation
    - gaps >= 3 days -> drop the observation
    - log-return r_t = ln(P_t / P_{t-1})
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

# NOTE ON PATH DEPTH: after the Luaran/ folder reorg, this module lives at
# Luaran/scripts/00_modul_inti/s2_data_pipeline.py -- three levels below Luaran/,
# so it takes three .parent hops (not two) to reach Luaran/data/raw.
RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


def load_raw_csv(asset_ticker: str) -> pd.DataFrame:
    """
    Loads the local CSV produced by 00_fetch_yfinance_LOCAL.py
    (data/raw/{asset}_yfinance_raw.csv), columns: date (index), close, volume.
    """
    path = RAW_DIR / f"{asset_ticker}_yfinance_raw.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 00_fetch_yfinance_LOCAL.py on a machine "
            f"with internet access and upload the resulting CSV into "
            f"data/raw/ before running this pipeline."
        )
    df = pd.read_csv(path, index_col="date", parse_dates=["date"])
    missing_cols = {"close", "volume"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"{path} is missing expected column(s): {missing_cols}")
    df = df[["close", "volume"]].sort_index()

    # UNIT FIX: Yahoo Finance's crypto Volume is USD-denominated, not
    # base-asset units. Convert immediately so everything downstream
    # (s3_monte_carlo.py, s4_inventory_policy.py, s5_backtest.py) is unit-consistent
    # -- see module docstring for the evidence this is actually needed.
    df["volume"] = df["volume"] / df["close"]

    return df


def preprocess(df: pd.DataFrame, max_gap_days: int = 3) -> pd.DataFrame:
    """
    Proposal 3.3, paragraph 2:
      - reindex to a full daily calendar
      - gaps < max_gap_days -> linear interpolation
      - gaps >= max_gap_days -> drop those rows entirely
      - compute log-return r_t = ln(P_t / P_{t-1}) on close price

    Unchanged logic from the original pipeline.
    """
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_index)
    df.index.name = "date"

    is_missing = df["close"].isna()
    run_id = (~is_missing).cumsum()
    gap_lengths = is_missing.groupby(run_id).transform("sum")

    interpolatable = is_missing & (gap_lengths < max_gap_days)
    df.loc[interpolatable, :] = np.nan
    df = df.interpolate(method="linear", limit=max_gap_days - 1, limit_area="inside")

    df = df.dropna(subset=["close", "volume"])

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["log_return"])
    return df


def load_asset_data(asset_ticker: str) -> pd.DataFrame:
    """Load the local yfinance CSV + preprocess for one asset ("BTC" or "ETH")."""
    raw = load_raw_csv(asset_ticker)
    return preprocess(raw)
