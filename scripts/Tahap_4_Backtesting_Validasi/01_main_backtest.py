"""
End-to-end pipeline matching proposal BAB 3 (Metode Penelitian), sections
3.3 through 3.6, for one asset at a time.

CHANGE LOG vs the original Colab pipeline:
  - No more live-fetch-with-silent-synthetic-fallback. Data always comes
    from the local yfinance CSV (data/raw/{asset}_yfinance_raw.csv,
    produced by 00_fetch_yfinance_LOCAL.py run on a machine with internet
    access). If that file is missing, this fails loudly -- it will NOT
    quietly substitute synthetic data and let you mistake it for a real
    result. Use --offline explicitly if you want a synthetic smoke test.

Usage (CLI):
    cd Luaran/scripts/Tahap_4_Backtesting_Validasi
    python 01_main_backtest.py --asset BTC
    python 01_main_backtest.py --asset ALL
    python 01_main_backtest.py --asset BTC --offline    # synthetic smoke test only
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Shared modules live one level up in 00_modul_inti/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "00_modul_inti"))

import numpy as np
import pandas as pd

import config
import data_pipeline
import monte_carlo
import inventory_policy
import backtest


def synthetic_price_volume(start_date: str, end_date: str, seed: int = 42,
                            s0: float = 30_000.0, mu: float = 0.0005,
                            sigma: float = 0.035, fat_tail_df: int = 3) -> pd.DataFrame:
    """
    OFFLINE FALLBACK ONLY - not part of the proposal's method. Used purely
    to smoke-test the pipeline code without real data. Do NOT use this
    for actual research results.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start_date, end_date, freq="D")
    n = len(idx)
    t_draws = rng.standard_t(df=fat_tail_df, size=n)
    returns = mu + sigma * t_draws / np.sqrt(fat_tail_df / (fat_tail_df - 2))
    close = s0 * np.exp(np.cumsum(returns))
    base_volume = 5_000 + 40_000 * np.abs(returns) / sigma
    noise = rng.lognormal(mean=0, sigma=0.3, size=n)
    volume = base_volume * noise
    df = pd.DataFrame({"close": close, "volume": volume}, index=idx)
    df.index.name = "date"
    return df


def load_data(asset: str, offline: bool = False) -> pd.DataFrame:
    if offline:
        print(f"[offline mode] generating synthetic price/volume for {asset} "
              f"- NOT real market data, pipeline smoke-test only.", file=sys.stderr)
        raw = synthetic_price_volume(config.CALIBRATION_START, config.BACKTEST_END,
                                      seed=hash(asset) % (2**31))
        return data_pipeline.preprocess(raw)
    return data_pipeline.load_asset_data(asset)


def process_asset(asset: str, df: pd.DataFrame,
                   calibration_start: str, calibration_end: str,
                   backtest_start: str, backtest_end: str) -> dict:
    """
    Runs proposal sections 3.4-3.6 on an already-loaded, already-preprocessed
    DataFrame `df` (must have columns: close, volume, log_return).
    """
    calib_mask = (df.index >= calibration_start) & (df.index <= calibration_end)
    calib = df.loc[calib_mask]
    backtest_mask = (df.index >= backtest_start) & (df.index <= backtest_end)

    if calib.empty or not backtest_mask.any():
        raise ValueError(f"Insufficient data for {asset} in the requested window: "
                          f"{len(calib)} calibration rows "
                          f"[{calibration_start} -> {calibration_end}], "
                          f"{backtest_mask.sum()} backtest rows "
                          f"[{backtest_start} -> {backtest_end}].")

    # --- 3.4 step 3: elasticity estimated once on the calibration window ---
    elasticity = monte_carlo.estimate_price_demand_elasticity(
        calib["log_return"].values, calib["volume"].values,
    )
    print(f"[{asset}] estimated price-demand elasticity (beta): {elasticity:.4f}")

    mc_params = monte_carlo.walk_forward_daily_params(
        log_returns=df["log_return"], volume=df["volume"],
        elasticity=elasticity, lead_time_days=config.LEAD_TIME_DAYS,
        n_scenarios=config.N_SCENARIOS, seed=config.RANDOM_SEED,
        min_history_days=len(calib),
        exponent_clip=config.EXPONENT_CLIP,
    )
    backtest_mc_params = mc_params.loc[mc_params.index.isin(df.index[backtest_mask])]

    if backtest_mc_params.empty:
        raise ValueError(
            f"No backtest days had enough preceding history to run the "
            f"Monte Carlo walk-forward simulation for {asset}."
        )

    total_clipped = int(backtest_mc_params["n_clipped"].sum())
    if total_clipped > 0:
        print(f"[{asset}] WARNING: exponent clip (±{config.EXPONENT_CLIP}) was hit "
              f"{total_clipped} time(s) across the backtest window's Monte Carlo "
              f"draws. This must be reported -- it means the clip is materially "
              f"bounding some demand scenarios, not just a theoretical safeguard.")

    realized_demand = df.loc[backtest_mc_params.index, "volume"]
    price_series = df["close"]
    initial_inventory = float(df.loc[calib.index, "volume"].tail(30).mean())

    proposed_result = inventory_policy.run_proposed_policy(
        realized_demand=realized_demand, mc_params=backtest_mc_params,
        initial_inventory=initial_inventory,
    )
    baseline_result = inventory_policy.run_fixed_buffer_policy(
        realized_demand=realized_demand, full_volume_history=df["volume"],
        initial_inventory=initial_inventory,
    )

    proposed_metrics = backtest.compute_metrics(proposed_result, price_series)
    baseline_metrics = backtest.compute_metrics(baseline_result, price_series)
    comparison = backtest.paired_comparison(proposed_result, baseline_result, price_series)

    return {
        "asset": asset,
        "elasticity": elasticity,
        "n_exponent_clipped": total_clipped,
        "proposed_metrics": proposed_metrics,
        "baseline_metrics": baseline_metrics,
        "comparison": comparison,
        "proposed_result": proposed_result,
        "baseline_result": baseline_result,
    }


def run_pipeline(asset: str, offline: bool = False) -> dict:
    df = load_data(asset, offline)
    return process_asset(
        asset, df,
        config.CALIBRATION_START, config.CALIBRATION_END,
        config.BACKTEST_START, config.BACKTEST_END,
    )


def print_report(result: dict) -> None:
    asset = result["asset"]
    pm, bm = result["proposed_metrics"], result["baseline_metrics"]
    cmp_ = result["comparison"]

    print(f"\n{'='*60}\n  BACKTEST REPORT: {asset}  ({cmp_['n_days_compared']} trading days)\n{'='*60}")
    print(f"{'Metric':<28}{'Proposed (MC)':>16}{'Baseline (fixed)':>18}")
    print(f"{'Stockout rate':<28}{pm.stockout_rate:>15.2%} {bm.stockout_rate:>17.2%}")
    print(f"{'Mean daily holding cost':<28}{pm.mean_daily_holding_cost:>16.2f}{bm.mean_daily_holding_cost:>18.2f}")
    print(f"{'Restock frequency':<28}{pm.restock_frequency:>15.2%} {bm.restock_frequency:>17.2%}")
    print(f"{'Total stockout amount':<28}{pm.total_stockout_amount:>16.1f}{bm.total_stockout_amount:>18.1f}")

    print(f"\n--- Paired t-test (alpha={config.SIGNIFICANCE_ALPHA}) ---")
    for name, key in [("Stockout rate", "stockout_rate"),
                       ("Holding cost", "holding_cost"),
                       ("Restock frequency (reported, not a criterion)", "restock_frequency")]:
        d = cmp_[key]
        print(f"{name}: t={d['t_statistic']:.3f}, p={d['p_value']:.4f}, "
              f"significant={d['significant_at_alpha']}, "
              f"proposed_lower={d['mean_diff_proposed_minus_baseline'] < 0}")

    print(f"\nExponent clip hits during backtest: {result['n_exponent_clipped']}")
    print(f"Verdict: {cmp_['verdict']}")


def metrics_summary_table(result: dict) -> pd.DataFrame:
    pm, bm, cmp_ = result["proposed_metrics"], result["baseline_metrics"], result["comparison"]
    rows = [
        {"metric": "stockout_rate", "proposed": pm.stockout_rate, "baseline": bm.stockout_rate,
         "p_value": cmp_["stockout_rate"]["p_value"],
         "significant": cmp_["stockout_rate"]["significant_at_alpha"],
         "proposed_better": cmp_["stockout_rate"]["proposed_significantly_lower"]},
        {"metric": "mean_daily_holding_cost", "proposed": pm.mean_daily_holding_cost,
         "baseline": bm.mean_daily_holding_cost,
         "p_value": cmp_["holding_cost"]["p_value"],
         "significant": cmp_["holding_cost"]["significant_at_alpha"],
         "proposed_better": cmp_["holding_cost"]["proposed_significantly_lower"]},
        {"metric": "restock_frequency", "proposed": pm.restock_frequency, "baseline": bm.restock_frequency,
         "p_value": cmp_["restock_frequency"]["p_value"],
         "significant": cmp_["restock_frequency"]["significant_at_alpha"],
         "proposed_better": None},
        {"metric": "total_stockout_amount", "proposed": pm.total_stockout_amount,
         "baseline": bm.total_stockout_amount, "p_value": None, "significant": None, "proposed_better": None},
    ]
    df = pd.DataFrame(rows)
    df.insert(0, "asset", result["asset"])
    df["elasticity"] = result["elasticity"]
    df["n_exponent_clipped"] = result["n_exponent_clipped"]
    return df


def export_results(results: dict, export_dir: str | None = None) -> dict:
    """
    BUG FIX: running `main.py --asset BTC` then `main.py --asset ETH`
    separately (as the instructions originally suggested, for readable
    per-asset logs) used to silently DROP the previous asset's row --
    each run overwrote metrics_summary.csv from scratch with only the
    assets in that run's `results` dict. Fixed: if metrics_summary.csv
    already exists, this now merges in the new run's rows (replacing any
    existing rows for the same asset) instead of truncating the file.
    Prefer `--asset ALL` in one run when possible; this fix exists so
    per-asset runs (for cleaner logs) don't lose data either way.

    PATH FIX (post folder-reorg): export_dir used to default to the
    CWD-relative string "../output/tables", which only resolved correctly
    if you ran this script from inside scripts/. Now nested three levels
    under Luaran/ (Luaran/scripts/Tahap_4_.../01_main_backtest.py), a
    relative string breaks depending on where you invoke it from. Default
    is now computed from this file's own location instead, so it resolves
    to Luaran/output/tables regardless of the caller's working directory.
    """
    import os
    if export_dir is None:
        export_dir = str(Path(__file__).resolve().parent.parent.parent / "output" / "tables")
    os.makedirs(export_dir, exist_ok=True)
    written = {"metrics_summary": None, "daily_series": []}

    summary_frames = [metrics_summary_table(result) for result in results.values()]
    if summary_frames:
        summary_path = os.path.join(export_dir, "metrics_summary.csv")
        new_summary = pd.concat(summary_frames, ignore_index=True)
        if os.path.exists(summary_path):
            existing = pd.read_csv(summary_path)
            assets_in_this_run = set(new_summary["asset"].unique())
            existing = existing[~existing["asset"].isin(assets_in_this_run)]
            new_summary = pd.concat([existing, new_summary], ignore_index=True)
        new_summary.to_csv(summary_path, index=False)
        written["metrics_summary"] = summary_path

    for asset, result in results.items():
        for label, key in (("proposed", "proposed_result"), ("baseline", "baseline_result")):
            path = os.path.join(export_dir, f"{asset}_{label}_daily.csv")
            result[key].to_csv(path)
            written["daily_series"].append(path)

    return written


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo SS/ROP inventory backtest (BTC/ETH CEX)")
    parser.add_argument("--asset", choices=config.ASSETS + ["ALL"], default="ALL")
    parser.add_argument("--offline", action="store_true",
                         help="use synthetic data instead of local yfinance CSVs (smoke test only)")
    args = parser.parse_args()

    assets = config.ASSETS if args.asset == "ALL" else [args.asset]
    all_results = {}
    for asset in assets:
        result = run_pipeline(asset, offline=args.offline)
        print_report(result)
        all_results[asset] = result

    written = export_results(all_results)
    print(f"\nExported: {written['metrics_summary']}")
    for p in written["daily_series"]:
        print(f"  {p}")
    return all_results


if __name__ == "__main__":
    main()
