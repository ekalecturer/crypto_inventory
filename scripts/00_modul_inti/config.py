"""
Configuration constants, matched 1:1 to proposal section 3.3-3.6.

Section references are to:
OPTIMASI PENGISIAN ULANG INVENTORI ASET KRIPTO PADA BURSA TERPUSAT
MENGGUNAKAN SIMULASI MONTE CARLO (ITK, 2026)

CHANGE LOG vs the original Colab pipeline (crypto_inventory_mc_colab.ipynb):
  - Data source switched from CoinGecko to yfinance (Binance blocked in
    Indonesia per logbook 30/04/2026; CoinGecko free tier capped at 365
    days, paid tier capped at 2 years -- neither covers the 2022-2024
    calibration window, which must include the FTX collapse). See
    docs/DATA_SPEC.md for the full rationale.
  - Added EXPONENT_CLIP to bound the elasticity scaling exponent in
    monte_carlo.py (see that module's docstring for why).
"""

# --- 3.3 Data collection ---
ASSETS = ["BTC", "ETH"]
YFINANCE_TICKERS = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
CALIBRATION_START = "2022-01-01"
CALIBRATION_END = "2024-12-31"
BACKTEST_START = "2025-01-01"
BACKTEST_END = "2025-12-31"
MAX_GAP_DAYS_FOR_INTERPOLATION = 3  # gaps <3 days: linear interpolate; >=3 days: drop

# --- 3.4 Monte Carlo simulation ---
N_SCENARIOS = 10_000                # N = 10,000 paths per proposal 3.4 step 2
LEAD_TIME_DAYS = 1                  # L = 1 trading day per proposal 3.5

# EXPONENT_CLIP: bounds |elasticity * r| before it goes into exp() in
# monte_carlo.simulate_demand_scenarios. Without this, a walk-forward
# expanding window that has absorbed a crisis-era return (e.g. an FTX-
# collapse-magnitude single-day move) combined with a non-trivial
# elasticity estimate can send exp(elasticity * r) to overflow, producing
# demand scenarios many orders of magnitude larger than anything
# physically plausible and making stockout/holding-cost metrics
# meaningless (this is exactly the ETH bug documented in
# Draft_Artikel_Preliminer_Monte_Carlo_Inventori_Kripto_rev.docx, Bagian
# 5). Clip value of 10 caps the multiplier at [e^-10, e^10] =
# [~0.00005x, ~22,026x] of baseline volume -- still a wide range, but
# finite and reproducible. This is a documented modeling assumption, not
# a value derived from the proposal text (the proposal does not specify
# one), and should be reported as such.
EXPONENT_CLIP = 10.0

# --- 3.5 Safety Stock / Reorder Point ---
SERVICE_LEVEL = 0.95                # 95% service level -> Z = 1.645
Z_SCORE = 1.645
TARGET_INVENTORY_PERCENTILE = 95    # target inventory = P95 of simulated L-day demand

# --- 3.6 Backtesting ---
BASELINE_ROLLING_WINDOW_DAYS = 30   # fixed buffer = % of 30-day rolling avg volume
BASELINE_BUFFER_PCT = 1.10          # buffer set at 110% of rolling mean (documented assumption,
                                    # proposal does not specify the exact %; must be justified/tuned)
SIGNIFICANCE_ALPHA = 0.05           # paired t-test at 5%

# --- Holding cost model (NOT specified numerically in proposal - assumption, flag it) ---
DAILY_HOLDING_COST_RATE = 0.0002    # 2 bps/day opportunity cost on capital tied up in inventory,
                                    # applied to (inventory_value). This is an explicit modeling
                                    # assumption standing in for the proposal's undefined "biaya holding".

RANDOM_SEED = 42
