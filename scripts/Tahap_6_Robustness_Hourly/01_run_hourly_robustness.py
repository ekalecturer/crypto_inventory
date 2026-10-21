"""
Tahap 6 -- Uji Ketahanan pada Frekuensi Data Lebih Tinggi (1 Jam).

TUJUAN (disepakati dengan Eka, 12 Agu 2026): ini BUKAN pengganti hasil
backtest harian resmi di Tabel 1 LKP -- angka harian tetap yang dilaporkan
sebagai hasil utama, tetap "final terkonfirmasi". Skrip ini adalah UJI
KETAHANAN TAMBAHAN (robustness check): mengulang metodologi Bab 3.3-3.6
yang sama persis pada data candlestick 1-jam, untuk melihat apakah
kesimpulan (kriteria superioritas TIDAK terpenuhi pada BTC maupun ETH)
tetap konsisten pada resolusi temporal yang lebih halus.

SUMBER DATA: CSV yang diunduh manual oleh Eka dari CryptoDataDownload.com
(Binance_BTCUSDT_1h.csv, Binance_ETHUSDT_1h.csv) -- BUKAN yfinance, dan
BUKAN dari s2_data_pipeline.py (modul itu hardcoded untuk kalender harian
dan format CSV yfinance). Loader di bawah ini ditulis khusus untuk file
CryptoDataDownload dan kalender per-jam.

PERBEDAAN PARAMETER vs BACKTEST HARIAN (didokumentasikan eksplisit, bukan
disembunyikan):
  - Satu "periode" di sini = 1 jam, bukan 1 hari.
  - Lead time L tetap merepresentasikan waktu tunggu FISIK 1 hari kalender
    -- pada resolusi jam, itu berarti L = 24 periode (bukan L = 1 seperti
    di s1_config.LEAD_TIME_DAYS). Ini pilihan yang disengaja supaya
    perbandingan tetap apple-to-apple secara fisik, bukan sekadar
    mengganti satuan tanpa makna.
  - Rolling window baseline 30 hari -> 30*24 = 720 periode jam.
  - DAILY_HOLDING_COST_RATE (2 bps/hari) dibagi 24 supaya tetap
    merepresentasikan biaya oportunitas yang sama per hari kalender,
    hanya diterapkan per-jam.
  - Gap threshold pra-pemrosesan 3 hari -> 72 jam (skala proporsional).
  - Elastisitas diestimasi ULANG dari data jam pada jendela kalibrasi yang
    sama (2022-2024) -- BUKAN dipakai ulang dari nilai harian, karena
    hubungan return-volume pada frekuensi jam adalah sinyal yang berbeda
    (lebih berisik, dipengaruhi microstructure) dan harus diestimasi
    sendiri, bukan diasumsikan sama.

Kompleksitas komputasi: walk-forward MC pada ~8.760 periode jam backtest
2025 x N=10.000 skenario x L=24 draw per skenario. Diuji berjalan dalam
hitungan menit (numpy vectorized per periode), bukan jam -- lihat log
runtime yang dicetak skrip ini.

Usage:
    cd Luaran/scripts/Tahap_6_Robustness_Hourly
    python 01_run_hourly_robustness.py
Reads:  Luaran/data/raw/Binance_{BTC,ETH}USDT_1h.csv
Writes: Luaran/output_hourly/tables/*.csv, Luaran/output_hourly/figures/langkahN_*.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Shared modules live in 00_modul_inti/, two levels up from this script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "00_modul_inti"))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import s1_config as config
import s3_monte_carlo as monte_carlo
import s4_inventory_policy as inventory_policy
import s5_backtest as backtest

# --- paths -------------------------------------------------------------
LUARAN_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = LUARAN_DIR / "data" / "raw"
OUT_DIR = LUARAN_DIR / "output_hourly"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

# --- hourly-specific parameters (documented deviations, see module docstring) ---
ASSET_FILES = {"BTC": "Binance_BTCUSDT_1h.csv", "ETH": "Binance_ETHUSDT_1h.csv"}
CALIBRATION_START = config.CALIBRATION_START   # 2022-01-01, unchanged
CALIBRATION_END = config.CALIBRATION_END       # 2024-12-31, unchanged
BACKTEST_START = config.BACKTEST_START         # 2025-01-01, unchanged
BACKTEST_END = config.BACKTEST_END             # 2025-12-31, unchanged
LEAD_TIME_PERIODS = 24                         # 1 calendar day of physical lead time, in hourly periods
MAX_GAP_HOURS = 72                             # 3 days, scaled from s1_config's 3-day rule
BASELINE_ROLLING_PERIODS = config.BASELINE_ROLLING_WINDOW_DAYS * 24   # 720
HOLDING_COST_RATE_PER_HOUR = config.DAILY_HOLDING_COST_RATE / 24
COLORS = {"rop": "#F2A65A", "ss": "#5B6472", "inv": "#1E2761", "hist": "#5B6472", "boot": "#F2A65A"}


# =========================================================================
# STEP 1 -- Data loading & pre-processing
# =========================================================================

def load_hourly_csv(asset: str) -> pd.DataFrame:
    path = RAW_DIR / ASSET_FILES[asset]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path, skiprows=1)
    # NOTE: the 'Unix' column has a data-source glitch (~672 rows per asset,
    # concentrated around 2025-03-11) where some values are in microseconds
    # instead of milliseconds -- confirmed by cross-checking against the
    # 'Date' string column, which does NOT have this problem. Use 'Date',
    # not 'Unix', for the timestamp. 'Date' itself has a mixed format
    # (some rows carry a '.000' fractional-second suffix) -- format='mixed'
    # handles both.
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    vol_col = f"Volume {asset}"
    df = df[["Date", "Close", vol_col]].rename(
        columns={"Date": "date", "Close": "close", vol_col: "volume"}
    )
    df = df.sort_values("date").set_index("date")
    n_dupes = int(df.index.duplicated().sum())
    if n_dupes:
        print(f"[{asset}] dropping {n_dupes} duplicate timestamp(s) (keep first)")
        df = df[~df.index.duplicated(keep="first")]
    return df


def preprocess_hourly(df: pd.DataFrame, start: str, end: str,
                       max_gap_hours: int = MAX_GAP_HOURS) -> tuple[pd.DataFrame, dict]:
    """Same rule as s2_data_pipeline.preprocess, scaled to hourly periods:
    gaps < max_gap_hours -> linear interpolation; gaps >= max_gap_hours -> drop.
    """
    full_index = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    df = df.reindex(full_index)
    df.index.name = "date"
    n_expected = len(full_index)

    is_missing = df["close"].isna()
    n_missing_raw = int(is_missing.sum())
    run_id = (~is_missing).cumsum()
    gap_lengths = is_missing.groupby(run_id).transform("sum")

    interpolatable = is_missing & (gap_lengths < max_gap_hours)
    n_interpolated = int(interpolatable.sum())
    df = df.interpolate(method="linear", limit=max_gap_hours - 1, limit_area="inside")
    df = df.dropna(subset=["close", "volume"])
    n_final = len(df)

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["log_return"])

    gap_report = {
        "n_periods_expected": n_expected,
        "n_periods_missing_raw": n_missing_raw,
        "n_periods_interpolated": n_interpolated,
        "n_periods_final_after_dropna": n_final,
        "pct_missing_raw": round(100 * n_missing_raw / n_expected, 3),
    }
    return df, gap_report


# =========================================================================
# STEP 2 -- KS validation + fat-tail distribution
# =========================================================================

def ks_validation(calib_returns: np.ndarray, n_scenarios: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    bootstrapped = rng.choice(calib_returns, size=n_scenarios, replace=True)
    ks_stat, ks_p = stats.ks_2samp(calib_returns, bootstrapped)
    hist_kurt = float(stats.kurtosis(calib_returns, fisher=True))
    boot_kurt = float(stats.kurtosis(bootstrapped, fisher=True))
    return {
        "n_historical": len(calib_returns),
        "n_bootstrapped": n_scenarios,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "ks_fails_to_reject_h0_at_0.05": bool(ks_p > 0.05),
        "historical_excess_kurtosis": hist_kurt,
        "bootstrapped_excess_kurtosis": boot_kurt,
        "both_fat_tailed": bool(hist_kurt > 0 and boot_kurt > 0),
    }, bootstrapped


def plot_fat_tail(asset: str, calib_returns: np.ndarray, bootstrapped: np.ndarray, ks: dict, step_num: int):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(calib_returns, bins=200, density=True, alpha=0.55, color=COLORS["hist"],
            label=f"Historis (n={ks['n_historical']:,})")
    ax.hist(bootstrapped, bins=200, density=True, alpha=0.45, color=COLORS["boot"],
            label=f"Bootstrap (n={ks['n_bootstrapped']:,})")
    ax.set_xlim(np.percentile(calib_returns, 0.5), np.percentile(calib_returns, 99.5))
    ax.set_title(f"{asset}: Distribusi Log-Return per Jam (Kalibrasi 2022-2024)\n"
                 f"KS p={ks['ks_pvalue']:.4f}, kurtosis hist={ks['historical_excess_kurtosis']:.2f}, "
                 f"boot={ks['bootstrapped_excess_kurtosis']:.2f}")
    ax.set_xlabel("Log-return per jam")
    ax.set_ylabel("Densitas")
    ax.legend()
    fig.tight_layout()
    out_path = FIG_DIR / f"langkah{step_num}_{asset}_distribusi_hourly.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{asset}] wrote {out_path}")
    return out_path


# =========================================================================
# STEP 3 -- SS/ROP walk-forward trend (recent window, for visualization)
# =========================================================================

def plot_ssrop_trend(asset: str, df: pd.DataFrame, elasticity: float, step_num: int,
                      trailing_periods: int = 720):
    tail = df.tail(trailing_periods)
    rng = np.random.default_rng(config.RANDOM_SEED)
    rows = []
    # subsample every 4th hour for the trend plot to keep runtime sane --
    # this is a VISUALIZATION only, not the backtest (backtest in step 4
    # runs every single hourly period, no subsampling).
    sample_dates = tail.index[::4]
    for d in sample_dates:
        hist = df.loc[:d]
        out = monte_carlo.simulate_demand_scenarios(
            historical_log_returns=hist["log_return"].values,
            historical_volume_mean=hist["volume"].mean(),
            elasticity=elasticity,
            lead_time_days=LEAD_TIME_PERIODS,
            n_scenarios=2000,   # reduced for plot responsiveness, same convention as the interactive dashboard
            rng=rng,
            exponent_clip=config.EXPONENT_CLIP,
        )
        ss, rop = inventory_policy.compute_ss_rop(out.sigma_d, out.d_bar, out.p95_demand,
                                                    lead_time=LEAD_TIME_PERIODS)
        rows.append({"date": d, "ss": ss, "rop": rop})
    trend = pd.DataFrame(rows).set_index("date")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(trend.index, trend["rop"], "--", color=COLORS["rop"], label="ROP")
    ax.plot(trend.index, trend["ss"], ":", color=COLORS["ss"], label="Safety Stock")
    ax.set_title(f"{asset}: Tren SS/ROP {trailing_periods}-Jam Terakhir (L={LEAD_TIME_PERIODS} periode = 1 hari)")
    ax.set_xlabel("Tanggal/jam")
    ax.set_ylabel("Unit aset")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = FIG_DIR / f"langkah{step_num}_{asset}_ssrop_hourly.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{asset}] wrote {out_path}")
    return out_path, trend


# =========================================================================
# STEP 4 -- Backtest (out-of-sample 2025, hourly)
# =========================================================================

def fast_walk_forward_hourly(df: pd.DataFrame, elasticity: float, lead_time: int,
                              n_scenarios: int, seed: int, min_history: int,
                              exponent_clip: float) -> pd.DataFrame:
    """
    Performance-optimized re-implementation of
    s3_monte_carlo.walk_forward_daily_params for the ~8,760-period hourly
    backtest. The shared function re-slices `.iloc[:i+1]` and recomputes
    `.mean()` over a growing pandas Series on EVERY one of ~8,760
    iterations -- O(n) per iteration, O(n^2) total, which is fine at daily
    scale (n~1,461) but far too slow at hourly scale (n~35,000): a naive
    port measured at ~20ms/period purely from that overhead, i.e. still
    only ~3 minutes for the full backtest, but this optimized version
    removes it entirely so the loop is dominated only by the actual Monte
    Carlo math (numpy array views instead of copies, cumulative sum
    instead of repeated .mean()) -- same formulas, same results, just
    without the accidental O(n^2).
    """
    log_returns_arr = df["log_return"].to_numpy()
    volume_arr = df["volume"].to_numpy()
    cumsum_vol = np.cumsum(volume_arr)
    dates = df.index
    n = len(df)

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(min_history - 1, n):
        n_hist = i + 1
        hist_returns = log_returns_arr[:n_hist]          # numpy view, O(1)
        hist_vol_mean = cumsum_vol[i] / n_hist            # O(1) via prefix sum
        out = monte_carlo.simulate_demand_scenarios(
            historical_log_returns=hist_returns,
            historical_volume_mean=hist_vol_mean,
            elasticity=elasticity,
            lead_time_days=lead_time,
            n_scenarios=n_scenarios,
            rng=rng,
            exponent_clip=exponent_clip,
        )
        rows.append({
            "date": dates[i], "sigma_d": out.sigma_d, "d_bar": out.d_bar,
            "p95_demand": out.p95_demand, "n_clipped": out.n_clipped,
        })
    return pd.DataFrame(rows).set_index("date")


def run_proposed_policy_hourly(realized_demand: pd.Series, mc_params: pd.DataFrame,
                                initial_inventory: float, lead_time: int) -> pd.DataFrame:
    """Same bookkeeping as s4_inventory_policy.run_proposed_policy, but with
    an explicit lead_time passed to compute_ss_rop -- the shared function's
    default argument is bound to config.LEAD_TIME_DAYS=1 at *module import
    time*, so it cannot be overridden by mutating config afterward. Rather
    than monkeypatch, this reimplements the same loop with lead_time made
    explicit, matching the shared module's logic exactly otherwise.
    """
    dates = realized_demand.index.intersection(mc_params.index)
    if len(dates) == 0:
        raise ValueError("No overlapping periods between realized_demand and mc_params.")
    inventory = initial_inventory
    rows = []
    for date in dates:
        sigma_d = mc_params.loc[date, "sigma_d"]
        d_bar = mc_params.loc[date, "d_bar"]
        target_inventory = mc_params.loc[date, "p95_demand"]
        ss, rop = inventory_policy.compute_ss_rop(sigma_d, d_bar, target_inventory, lead_time=lead_time)

        inv_start = inventory
        reorder_triggered = inv_start <= rop
        reorder_qty = max(0.0, target_inventory - inv_start + ss) if reorder_triggered else 0.0
        inv_after_reorder = inv_start + reorder_qty

        demand_now = float(realized_demand.loc[date])
        stockout = demand_now > inv_after_reorder
        stockout_amount = max(0.0, demand_now - inv_after_reorder)
        inv_end = max(0.0, inv_after_reorder - demand_now)

        rows.append({
            "date": date, "inventory_start": inv_start, "demand_realized": demand_now,
            "reorder_triggered": reorder_triggered, "reorder_qty": reorder_qty,
            "stockout": stockout, "stockout_amount": stockout_amount, "inventory_end": inv_end,
            "ss": ss, "rop": rop, "target_inventory": target_inventory,
        })
        inventory = inv_end
    return pd.DataFrame(rows).set_index("date")


def plot_backtest(asset: str, proposed: pd.DataFrame, baseline: pd.DataFrame,
                   proposed_metrics, baseline_metrics, step_num: int):
    # Trajectory plot: 8,760 hourly points is unreadable -- resample to
    # daily mean for visualization only (the backtest itself, and all
    # reported metrics, use every hourly period -- no subsampling there).
    prop_daily = proposed[["inventory_end", "ss", "rop"]].resample("D").mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prop_daily.index, prop_daily["inventory_end"], color=COLORS["inv"], label="Inventori (proposed)")
    ax.plot(prop_daily.index, prop_daily["rop"], "--", color=COLORS["rop"], label="ROP")
    ax.plot(prop_daily.index, prop_daily["ss"], ":", color=COLORS["ss"], label="Safety Stock")
    ax.set_title(f"{asset}: Trayektori Inventori vs SS/ROP, Backtest 2025 (rata-rata harian dari data jam)")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Unit aset")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    traj_path = FIG_DIR / f"langkah{step_num}_{asset}_trajectory_hourly.png"
    fig.savefig(traj_path, dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    metrics_names = ["Stockout rate", "Restock frequency"]
    prop_vals = [proposed_metrics.stockout_rate, proposed_metrics.restock_frequency]
    base_vals = [baseline_metrics.stockout_rate, baseline_metrics.restock_frequency]
    x = np.arange(len(metrics_names))
    width = 0.35
    ax2.bar(x - width / 2, prop_vals, width, label="Proposed (MC)", color=COLORS["inv"])
    ax2.bar(x + width / 2, base_vals, width, label="Baseline (fixed)", color=COLORS["rop"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_names)
    ax2.set_ylabel("Proporsi periode jam")
    ax2.set_title(f"{asset}: Perbandingan Metrik, Backtest Per-Jam 2025")
    ax2.legend()
    fig2.tight_layout()
    metrics_path = FIG_DIR / f"langkah{step_num}_{asset}_metrics_hourly.png"
    fig2.savefig(metrics_path, dpi=150)
    plt.close(fig2)

    print(f"[{asset}] wrote {traj_path}")
    print(f"[{asset}] wrote {metrics_path}")
    return traj_path, metrics_path


# =========================================================================
# Orchestration
# =========================================================================

def process_asset(asset: str) -> dict:
    t0 = time.time()
    print(f"\n{'='*70}\n  {asset}: STEP 1 -- load & pre-process hourly data\n{'='*70}")
    raw = load_hourly_csv(asset)
    df, gap_report = preprocess_hourly(raw, CALIBRATION_START, BACKTEST_END)
    print(f"[{asset}] gap report: {gap_report}")

    calib_mask = (df.index >= CALIBRATION_START) & (df.index <= CALIBRATION_END)
    calib = df.loc[calib_mask]
    backtest_mask = (df.index >= BACKTEST_START) & (df.index <= BACKTEST_END)
    print(f"[{asset}] calibration periods: {len(calib):,} | backtest periods: {int(backtest_mask.sum()):,}")

    print(f"\n{'='*70}\n  {asset}: STEP 2 -- KS validation + fat-tail distribution\n{'='*70}")
    ks, bootstrapped = ks_validation(calib["log_return"].values, n_scenarios=config.N_SCENARIOS, seed=config.RANDOM_SEED)
    print(f"[{asset}] KS: {ks}")
    plot_fat_tail(asset, calib["log_return"].values, bootstrapped, ks, step_num=2)

    elasticity = monte_carlo.estimate_price_demand_elasticity(calib["log_return"].values, calib["volume"].values)
    print(f"[{asset}] elasticity (hourly re-estimate): {elasticity:.4f}")

    print(f"\n{'='*70}\n  {asset}: STEP 3 -- SS/ROP walk-forward trend (last 720h)\n{'='*70}")
    plot_ssrop_trend(asset, df, elasticity, step_num=3)

    print(f"\n{'='*70}\n  {asset}: STEP 4 -- backtest 2025 (hourly, every period)\n{'='*70}")
    t_bt0 = time.time()
    mc_params = fast_walk_forward_hourly(
        df=df, elasticity=elasticity, lead_time=LEAD_TIME_PERIODS,
        n_scenarios=config.N_SCENARIOS, seed=config.RANDOM_SEED,
        min_history=len(calib),
        exponent_clip=config.EXPONENT_CLIP,
    )
    backtest_mc_params = mc_params.loc[mc_params.index.isin(df.index[backtest_mask])]
    if backtest_mc_params.empty:
        raise ValueError(f"[{asset}] no backtest periods had enough history for walk-forward MC.")
    total_clipped = int(backtest_mc_params["n_clipped"].sum())
    print(f"[{asset}] walk-forward MC over {len(backtest_mc_params):,} backtest periods took "
          f"{time.time()-t_bt0:.1f}s; exponent clip hits: {total_clipped}")

    realized_demand = df.loc[backtest_mc_params.index, "volume"]
    price_series = df["close"]
    initial_inventory = float(df.loc[calib.index, "volume"].tail(BASELINE_ROLLING_PERIODS).mean())

    proposed_result = run_proposed_policy_hourly(
        realized_demand=realized_demand, mc_params=backtest_mc_params,
        initial_inventory=initial_inventory, lead_time=LEAD_TIME_PERIODS,
    )
    baseline_result = inventory_policy.run_fixed_buffer_policy(
        realized_demand=realized_demand, full_volume_history=df["volume"],
        initial_inventory=initial_inventory,
        rolling_window=BASELINE_ROLLING_PERIODS, buffer_pct=config.BASELINE_BUFFER_PCT,
    )

    proposed_metrics = backtest.compute_metrics(proposed_result, price_series, holding_cost_rate=HOLDING_COST_RATE_PER_HOUR)
    baseline_metrics = backtest.compute_metrics(baseline_result, price_series, holding_cost_rate=HOLDING_COST_RATE_PER_HOUR)
    comparison = backtest.paired_comparison(proposed_result, baseline_result, price_series,
                                             holding_cost_rate=HOLDING_COST_RATE_PER_HOUR)

    plot_backtest(asset, proposed_result, baseline_result, proposed_metrics, baseline_metrics, step_num=4)

    print(f"[{asset}] TOTAL runtime: {time.time()-t0:.1f}s")
    return {
        "asset": asset, "gap_report": gap_report, "ks": ks, "elasticity": elasticity,
        "n_exponent_clipped": total_clipped,
        "proposed_metrics": proposed_metrics, "baseline_metrics": baseline_metrics,
        "comparison": comparison, "proposed_result": proposed_result, "baseline_result": baseline_result,
    }


def export_tables(results: dict):
    ks_rows = []
    metrics_rows = []
    for asset, r in results.items():
        ks_rows.append({"asset": asset, **r["ks"]})
        pm, bm, cmp_ = r["proposed_metrics"], r["baseline_metrics"], r["comparison"]
        for metric, p_val, b_val, sig_key in [
            ("stockout_rate", pm.stockout_rate, bm.stockout_rate, "stockout_rate"),
            ("mean_hourly_holding_cost", pm.mean_daily_holding_cost, bm.mean_daily_holding_cost, "holding_cost"),
            ("restock_frequency", pm.restock_frequency, bm.restock_frequency, "restock_frequency"),
        ]:
            metrics_rows.append({
                "asset": asset, "metric": metric, "proposed": p_val, "baseline": b_val,
                "p_value": cmp_[sig_key]["p_value"], "significant": cmp_[sig_key]["significant_at_alpha"],
                "proposed_better": cmp_[sig_key].get("proposed_significantly_lower"),
                "elasticity": r["elasticity"], "n_exponent_clipped": r["n_exponent_clipped"],
            })
        metrics_rows.append({
            "asset": asset, "metric": "total_stockout_amount", "proposed": pm.total_stockout_amount,
            "baseline": bm.total_stockout_amount, "p_value": None, "significant": None, "proposed_better": None,
            "elasticity": r["elasticity"], "n_exponent_clipped": r["n_exponent_clipped"],
        })
        r["proposed_result"].to_csv(TBL_DIR / f"{asset}_proposed_hourly.csv")
        r["baseline_result"].to_csv(TBL_DIR / f"{asset}_baseline_hourly.csv")

    pd.DataFrame(ks_rows).to_csv(TBL_DIR / "hourly_ks_validation.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(TBL_DIR / "hourly_metrics_summary.csv", index=False)
    print(f"\nWrote {TBL_DIR / 'hourly_ks_validation.csv'}")
    print(f"Wrote {TBL_DIR / 'hourly_metrics_summary.csv'}")


CHECKPOINT_DIR = OUT_DIR / "_checkpoints"


def main():
    """
    Supports two invocation modes because the full per-asset run (~75-90s
    dominated by the O(8,760-period) walk-forward Monte Carlo backtest)
    is long enough to be worth checkpointing per asset rather than always
    re-running both assets back-to-back in one process:

        python 01_run_hourly_robustness.py --asset BTC   # process + pickle checkpoint
        python 01_run_hourly_robustness.py --asset ETH   # process + pickle checkpoint
        python 01_run_hourly_robustness.py --finalize     # load both checkpoints, export tables

    Plain `python 01_run_hourly_robustness.py` with no args still runs
    both assets end-to-end and exports tables in one call, for anyone
    re-running this later without the same time constraint.
    """
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=config.ASSETS, default=None)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if args.finalize:
        results = {}
        for asset in config.ASSETS:
            ckpt = CHECKPOINT_DIR / f"{asset}.pkl"
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing checkpoint {ckpt} -- run --asset {asset} first.")
            with open(ckpt, "rb") as f:
                results[asset] = pickle.load(f)
        print(f"\n{'='*70}\n  RINGKASAN UJI KETAHANAN PER-JAM\n{'='*70}")
        for asset, r in results.items():
            print(f"\n{asset}: {r['comparison']['verdict']}")
        export_tables(results)
        return

    if args.asset:
        result = process_asset(args.asset)
        with open(CHECKPOINT_DIR / f"{args.asset}.pkl", "wb") as f:
            pickle.dump(result, f)
        print(f"\nCheckpoint saved: {CHECKPOINT_DIR / f'{args.asset}.pkl'}")
        return

    results = {}
    for asset in config.ASSETS:
        results[asset] = process_asset(asset)
    print(f"\n{'='*70}\n  RINGKASAN UJI KETAHANAN PER-JAM\n{'='*70}")
    for asset, r in results.items():
        print(f"\n{asset}: {r['comparison']['verdict']}")
    export_tables(results)


if __name__ == "__main__":
    main()
