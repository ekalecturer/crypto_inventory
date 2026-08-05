"""
RUN THIS ON YOUR OWN MACHINE (or Colab) -- NOT inside the Cowork sandbox.

The Cowork sandbox's outbound network does not reach query1.finance.yahoo.com
or api.coingecko.com (both return connection failures / 403 at the proxy).
This script has to be run somewhere with real internet access, and its
output CSVs uploaded back into data/raw/.

What it does
------------
Fetches daily OHLCV for BTC-USD and ETH-USD from Yahoo Finance via the
`yfinance` package, for 2022-01-01 through 2025-12-31 (covers both the
2022-2024 calibration window and the 2025 out-of-sample backtest window
per the proposal). Saves one CSV per asset to data/raw/, plus a short gap
report printed to stdout -- copy that output into the chat so gaps can be
documented in the Fase 0 summary (per proposal Bab 3.3: gaps < 3 days get
linearly interpolated downstream, gaps >= 3 days get dropped).

Setup
-----
    pip install yfinance pandas

Usage
-----
    python 00_fetch_yfinance_LOCAL.py
    # writes:
    #   data/raw/BTC_yfinance_raw.csv
    #   data/raw/ETH_yfinance_raw.csv

Then upload both CSVs back into this project's data/raw/ folder.
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Run: pip install yfinance pandas", file=sys.stderr)
    sys.exit(1)

TICKERS = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
START = "2022-01-01"
END = "2026-01-01"  # yfinance end date is exclusive; this gets through 2025-12-31
# NOTE ON PATH DEPTH: lives at Luaran/scripts/Tahap_1_.../01_fetch_yfinance_LOCAL.py
# -- three levels below Luaran/, so three .parent hops to reach Luaran/data/raw.
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


def fetch_one(asset: str, ticker: str) -> pd.DataFrame:
    print(f"\nFetching {asset} ({ticker}) from Yahoo Finance, {START} to 2025-12-31...")
    df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(
            f"yfinance returned no data for {ticker}. Check your internet "
            f"connection and that Yahoo Finance is reachable from this machine."
        )
    # yfinance sometimes returns a MultiIndex on columns for a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={"Close": "close", "Volume": "volume"})
    df = df[["close", "volume"]].copy()
    df.index.name = "date"
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()

    # gap report: how many calendar days are missing from the full daily range
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing = full_range.difference(df.index)
    print(f"  Retrieved {len(df)} trading days, {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  Missing calendar days (weekends included -- crypto trades 24/7, "
          f"so gaps here are real data gaps, not market closures): {len(missing)}")
    if len(missing) > 0:
        # summarize gap run-lengths
        missing_sorted = sorted(missing)
        runs = []
        run_start = missing_sorted[0]
        prev = missing_sorted[0]
        for d in missing_sorted[1:]:
            if (d - prev).days > 1:
                runs.append((run_start, prev))
                run_start = d
            prev = d
        runs.append((run_start, prev))
        print(f"  Gap runs ({len(runs)} total):")
        for r_start, r_end in runs:
            length = (r_end - r_start).days + 1
            print(f"    {r_start.date()} -> {r_end.date()}  ({length} day(s))"
                  f"{'  <-- will be DROPPED downstream (>=3 days)' if length >= 3 else '  <-- will be interpolated downstream (<3 days)'}")
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for asset, ticker in TICKERS.items():
        df = fetch_one(asset, ticker)
        out_path = OUT_DIR / f"{asset}_yfinance_raw.csv"
        df.to_csv(out_path)
        print(f"  Wrote {out_path}")

    print("\nDone. Upload the CSVs in data/raw/ back to the project, then continue "
          "with the Tahap 4 pipeline (Tahap_4_Backtesting_Validasi/01_main_backtest.py).")


if __name__ == "__main__":
    main()
