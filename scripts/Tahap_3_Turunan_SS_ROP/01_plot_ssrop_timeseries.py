"""
Tahap 3 -- Grafik Parameter Safety Stock dan Reorder Point.

Menghasilkan, per aset (BTC, ETH):
  1. Time series SS_t dan ROP_t sepanjang periode backtest 2025, hasil
     derivasi harian dari walk-forward Monte Carlo (bukti visual bahwa
     parameter bersifat dinamis, bukan konstan -- sesuai proposal 3.4
     paragraf terakhir).

Output: output/figures/tahap3_{asset}_ssrop_timeseries.png
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "00_modul_inti"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import data_pipeline
import monte_carlo
import inventory_policy

# Three .parent hops: this file -> Tahap_3_Turunan_SS_ROP -> scripts -> Luaran
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAVY = "#1E2761"
ACCENT = "#F2A65A"
GREY = "#5B6472"


def plot_asset(asset: str):
    df = data_pipeline.load_asset_data(asset)
    calib_mask = (df.index >= config.CALIBRATION_START) & (df.index <= config.CALIBRATION_END)
    calib = df.loc[calib_mask]
    backtest_mask = (df.index >= config.BACKTEST_START) & (df.index <= config.BACKTEST_END)

    elasticity = monte_carlo.estimate_price_demand_elasticity(
        calib["log_return"].values, calib["volume"].values,
    )
    mc_params = monte_carlo.walk_forward_daily_params(
        log_returns=df["log_return"], volume=df["volume"],
        elasticity=elasticity, lead_time_days=config.LEAD_TIME_DAYS,
        n_scenarios=config.N_SCENARIOS, seed=config.RANDOM_SEED,
        min_history_days=len(calib),
        exponent_clip=config.EXPONENT_CLIP,
    )
    backtest_mc_params = mc_params.loc[mc_params.index.isin(df.index[backtest_mask])]
    ss_series, rop_series = [], []
    for date in backtest_mc_params.index:
        sigma_d = backtest_mc_params.loc[date, "sigma_d"]
        d_bar = backtest_mc_params.loc[date, "d_bar"]
        target = backtest_mc_params.loc[date, "p95_demand"]
        ss, rop = inventory_policy.compute_ss_rop(sigma_d, d_bar, target)
        ss_series.append(ss)
        rop_series.append(rop)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(backtest_mc_params.index, rop_series, color=NAVY, linewidth=1.6, label="ROP_t (Reorder Point)")
    ax.fill_between(backtest_mc_params.index, 0, ss_series, color=ACCENT, alpha=0.5, label="SS_t (Safety Stock)")
    ax.set_title(f"{asset}: Parameter SS/ROP Harian, Backtest 2025 (walk-forward)", fontsize=12)
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Unit aset (basis harian)")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = OUT_DIR / f"tahap3_{asset}_ssrop_timeseries.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{asset}] wrote {out_path}  (elasticity beta={elasticity:.4f})")


def main():
    for asset in config.ASSETS:
        plot_asset(asset)


if __name__ == "__main__":
    main()
