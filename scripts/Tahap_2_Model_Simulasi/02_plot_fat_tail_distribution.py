"""
Tahap 2 -- Grafik Model Simulasi Monte Carlo.

Menghasilkan, per aset (BTC, ETH):
  1. Histogram overlay: distribusi log-return historis (kalibrasi) vs
     distribusi log-return hasil bootstrap -- bukti visual untuk indikator
     capaian Tahap 2 ("distribusi output mencerminkan karakteristik
     fat-tail" + uji Kolmogorov-Smirnov, lihat ks_validation.py).

Output: output/figures/tahap2_{asset}_fat_tail.png
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "00_modul_inti"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import config
import data_pipeline

# Three .parent hops: this file -> Tahap_2_Model_Simulasi -> scripts -> Luaran
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAVY = "#1E2761"
ACCENT = "#F2A65A"
GREY = "#5B6472"


def plot_asset(asset: str):
    df = data_pipeline.load_asset_data(asset)
    calib_mask = (df.index >= config.CALIBRATION_START) & (df.index <= config.CALIBRATION_END)
    historical_returns = df.loc[calib_mask, "log_return"].values

    rng = np.random.default_rng(config.RANDOM_SEED)
    draws = rng.choice(historical_returns, size=(config.N_SCENARIOS, config.LEAD_TIME_DAYS), replace=True)
    bootstrapped_returns = draws.flatten()

    ks_stat, ks_pvalue = stats.ks_2samp(historical_returns, bootstrapped_returns)
    hist_kurt = stats.kurtosis(historical_returns, fisher=True)
    boot_kurt = stats.kurtosis(bootstrapped_returns, fisher=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(
        min(historical_returns.min(), bootstrapped_returns.min()),
        max(historical_returns.max(), bootstrapped_returns.max()),
        80,
    )
    ax.hist(historical_returns, bins=bins, density=True, alpha=0.55, color=NAVY,
            label=f"Historis (kalibrasi, n={len(historical_returns)})")
    ax.hist(bootstrapped_returns, bins=bins, density=True, alpha=0.45, color=ACCENT,
            label=f"Bootstrap MC (n={len(bootstrapped_returns):,})")

    # overlay normal distribution for visual fat-tail contrast
    x = np.linspace(bins[0], bins[-1], 300)
    normal_pdf = stats.norm.pdf(x, historical_returns.mean(), historical_returns.std())
    ax.plot(x, normal_pdf, color=GREY, linestyle="--", linewidth=1.5, label="Normal (referensi)")

    ax.set_title(f"{asset}: Distribusi Log-Return Historis vs Bootstrap MC\n"
                 f"Uji KS: statistik={ks_stat:.4f}, p={ks_pvalue:.4f}  |  "
                 f"Excess kurtosis: historis={hist_kurt:.2f}, bootstrap={boot_kurt:.2f}",
                 fontsize=11)
    ax.set_xlabel("Log-return harian")
    ax.set_ylabel("Densitas")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_path = OUT_DIR / f"tahap2_{asset}_fat_tail.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{asset}] wrote {out_path} (KS p={ks_pvalue:.4f}, kurtosis hist={hist_kurt:.2f}/boot={boot_kurt:.2f})")


def main():
    for asset in config.ASSETS:
        plot_asset(asset)


if __name__ == "__main__":
    main()
