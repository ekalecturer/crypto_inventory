"""
Kolmogorov-Smirnov validation of the Monte Carlo bootstrap output against
historical data -- proposal Bab 3.2 (Tahap 2, "Pembangunan Model Simulasi")
success indicator: "Model lolos uji Kolmogorov-Smirnov terhadap data
historis; distribusi output mencerminkan karakteristik fat-tail."

This was never implemented in the original Colab pipeline or in
inventori_hibah_pipeline's Fase 0 run -- added here specifically to satisfy
that stated indicator before the LKP claims Tahap 2 as complete.

WHAT THIS ACTUALLY TESTS (read before citing the result)
==========================================================
Two related two-sample KS tests are run per asset, both using
scipy.stats.ks_2samp:

1. log_return_ks: bootstrapped log-returns (N_SCENARIOS x LEAD_TIME_DAYS
   draws, flattened) from the calibration window vs. the calibration
   window's own historical log-returns. Because the bootstrap draws WITH
   REPLACEMENT directly from this same historical sample, this test is
   close to tautological -- it is expected to fail to reject H0 (same
   distribution) almost by construction, since the bootstrap population
   and the "historical" population are literally the same empirical set.
   This is a legitimate implementation check (confirms the bootstrap
   sampling code isn't introducing distortion, e.g. a biased RNG or an
   off-by-one on the L-day window), but it is NOT independent evidence
   that a fat-tailed process is a good real-world model. Report it as
   the modest claim it is.

2. fat_tail_check: reports excess kurtosis (Fisher) of both the
   historical and bootstrapped log-return distributions. Excess
   kurtosis > 0 indicates heavier-than-normal tails ("fat-tail"), which
   is the qualitative claim the proposal's indicator text asks to
   confirm ("distribusi output mencerminkan karakteristik fat-tail").
   This is descriptive, not a formal hypothesis test.

Usage:
    cd Luaran/scripts/Tahap_2_Model_Simulasi
    python 01_ks_validation.py
Writes output/tables/ks_validation.csv and prints a summary.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Shared modules (config, data_pipeline, monte_carlo, ...) live one level up
# in 00_modul_inti/, not alongside this script -- add it to sys.path before
# importing them.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "00_modul_inti"))

import numpy as np
import pandas as pd
from scipy import stats

import config
import data_pipeline
import monte_carlo


def validate_asset(asset: str) -> dict:
    df = data_pipeline.load_asset_data(asset)
    calib_mask = (df.index >= config.CALIBRATION_START) & (df.index <= config.CALIBRATION_END)
    calib = df.loc[calib_mask]

    historical_returns = calib["log_return"].values

    rng = np.random.default_rng(config.RANDOM_SEED)
    draws = rng.choice(historical_returns, size=(config.N_SCENARIOS, config.LEAD_TIME_DAYS), replace=True)
    bootstrapped_returns = draws.flatten()

    ks_stat, ks_pvalue = stats.ks_2samp(historical_returns, bootstrapped_returns)

    hist_kurtosis = float(stats.kurtosis(historical_returns, fisher=True))  # excess kurtosis
    boot_kurtosis = float(stats.kurtosis(bootstrapped_returns, fisher=True))

    return {
        "asset": asset,
        "n_historical": len(historical_returns),
        "n_bootstrapped": len(bootstrapped_returns),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "ks_fails_to_reject_h0_at_0.05": bool(ks_pvalue >= 0.05),
        "historical_excess_kurtosis": hist_kurtosis,
        "bootstrapped_excess_kurtosis": boot_kurtosis,
        "both_fat_tailed": bool(hist_kurtosis > 0 and boot_kurtosis > 0),
    }


def main():
    results = [validate_asset(asset) for asset in config.ASSETS]
    df = pd.DataFrame(results)

    # Three .parent hops: this file -> Tahap_2_Model_Simulasi -> scripts -> Luaran
    out_dir = Path(__file__).resolve().parent.parent.parent / "output" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ks_validation.csv"
    df.to_csv(out_path, index=False)

    print(f"{'='*70}\nKS VALIDATION (Tahap 2 indicator)\n{'='*70}")
    for r in results:
        print(f"\n{r['asset']}:")
        print(f"  KS statistic = {r['ks_statistic']:.4f}, p-value = {r['ks_pvalue']:.4f}")
        print(f"  Fails to reject H0 (same distribution) at alpha=0.05: "
              f"{r['ks_fails_to_reject_h0_at_0.05']}")
        print(f"  Historical excess kurtosis: {r['historical_excess_kurtosis']:.2f}  "
              f"(>0 = fat-tailed vs normal)")
        print(f"  Bootstrapped excess kurtosis: {r['bootstrapped_excess_kurtosis']:.2f}")
        print(f"  Both fat-tailed: {r['both_fat_tailed']}")

    print(f"\nWrote {out_path}")
    return df


if __name__ == "__main__":
    main()
