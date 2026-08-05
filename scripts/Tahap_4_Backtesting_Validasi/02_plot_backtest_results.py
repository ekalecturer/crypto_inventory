"""
Tahap 4 -- Grafik Backtesting dan Validasi.

Menghasilkan, per aset (BTC, ETH):
  1. Trajektori inventori vs SS/ROP sepanjang backtest 2025 (kebijakan
     proposed), dengan penanda hari stockout -- versi grafis dari apa
     yang sebelumnya hanya ada sebagai angka pada Tabel 1 LKP.
  2. Bar chart perbandingan proposed vs baseline untuk tiga metrik
     (tingkat stockout, biaya holding, frekuensi pengisian ulang),
     dengan anotasi signifikansi (p-value) per metrik.

Output:
  output/figures/tahap4_{asset}_inventory_trajectory.png
  output/figures/tahap4_{asset}_metrics_comparison.png
"""

from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "00_modul_inti"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

# IMPORT NOTE: this used to be `import main as pipeline_main`. After the
# folder reorg, the backtest driver was renamed 01_main_backtest.py to
# reflect its step order within Tahap 4 -- but Python module names can't
# start with a digit, so `import 01_main_backtest` is a syntax error.
# Loading it by file path via importlib sidesteps that, while keeping the
# numeric-prefix naming convention used across every Tahap folder.
_main_path = Path(__file__).resolve().parent / "01_main_backtest.py"
_spec = importlib.util.spec_from_file_location("pipeline_main", _main_path)
pipeline_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline_main)

# Three .parent hops: this file -> Tahap_4_Backtesting_Validasi -> scripts -> Luaran
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAVY = "#1E2761"
ACCENT = "#F2A65A"
RED = "#B3261E"
GREEN = "#1B7A43"
GREY = "#5B6472"


def plot_trajectory(asset: str, result: dict):
    df = result["proposed_result"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df.index, df["inventory_start"], color=NAVY, linewidth=1.4, label="Inventori (awal hari)")
    ax.plot(df.index, df["rop"], color=ACCENT, linewidth=1.2, linestyle="--", label="ROP")
    ax.plot(df.index, df["ss"], color=GREY, linewidth=1.0, linestyle=":", label="Safety Stock")

    stockout_days = df.index[df["stockout"]]
    if len(stockout_days) > 0:
        ax.scatter(stockout_days, df.loc[stockout_days, "inventory_start"],
                   color=RED, s=14, zorder=5, label=f"Hari stockout (n={len(stockout_days)})")

    ax.set_title(f"{asset}: Trajektori Inventori vs SS/ROP -- Kebijakan Proposed, Backtest 2025", fontsize=12)
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Unit aset")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = OUT_DIR / f"tahap4_{asset}_inventory_trajectory.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{asset}] wrote {out_path}")


def plot_metrics_comparison(asset: str, result: dict):
    pm, bm = result["proposed_metrics"], result["baseline_metrics"]
    cmp_ = result["comparison"]

    metrics = [
        ("Tingkat Stockout", pm.stockout_rate * 100, bm.stockout_rate * 100, "%",
         cmp_["stockout_rate"]["p_value"], cmp_["stockout_rate"]["proposed_significantly_lower"]),
        ("Biaya Holding Harian\n(rata-rata)", pm.mean_daily_holding_cost, bm.mean_daily_holding_cost, "",
         cmp_["holding_cost"]["p_value"], cmp_["holding_cost"]["proposed_significantly_lower"]),
        ("Frekuensi Pengisian Ulang", pm.restock_frequency * 100, bm.restock_frequency * 100, "%",
         cmp_["restock_frequency"]["p_value"], None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (label, prop_val, base_val, unit, p_val, better) in zip(axes, metrics):
        bars = ax.bar(["Proposed\n(MC)", "Baseline\n(fixed)"], [prop_val, base_val],
                       color=[NAVY, GREY], width=0.55)
        for b, v in zip(bars, [prop_val, base_val]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.2f}{unit}",
                    ha="center", va="bottom", fontsize=9)
        sig_txt = f"p={p_val:.4f}" if p_val is not None else "p=n/a"
        if better is True:
            sig_txt += "  (proposed lebih baik)"
            color = GREEN
        elif better is False:
            sig_txt += "  (proposed lebih buruk)"
            color = RED
        else:
            color = GREY
        ax.set_title(label, fontsize=10)
        ax.text(0.5, -0.18, sig_txt, transform=ax.transAxes, ha="center", fontsize=8.5, color=color)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"{asset}: Proposed vs Baseline, Backtest 2025 ({cmp_['n_days_compared']} hari)", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = OUT_DIR / f"tahap4_{asset}_metrics_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{asset}] wrote {out_path}")


def main():
    for asset in config.ASSETS:
        result = pipeline_main.run_pipeline(asset, offline=False)
        plot_trajectory(asset, result)
        plot_metrics_comparison(asset, result)


if __name__ == "__main__":
    main()
