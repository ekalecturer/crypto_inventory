"""
Tahap 5 -- Generator data untuk dasbor (arsitektur GitHub Actions + JSON statis).

Berbeda dari main.py (yang menjalankan backtest penuh pada data historis),
skrip ini menjawab pertanyaan operasional harian yang dijelaskan proposal
Bab 3.7: "operator memasukkan level inventori saat ini -> dasbor
menampilkan SS terkini, ROP terkini, rekomendasi pengisian ulang, dan tren
30 hari terakhir."

Dirancang untuk dijalankan di GitHub Actions (punya akses internet penuh,
tidak seperti sandbox Cowork), BUKAN di lingkungan pengembangan ini.
Alur:
    1. Ambil ~400 hari data BTC-USD/ETH-USD terbaru via yfinance
       (cukup untuk walk-forward dengan min_history_days yang wajar).
    2. Hitung SS/ROP untuk HARI TERAKHIR yang tersedia (snapshot "hari ini").
    3. Simpan hasil + tren 30 hari ke data/latest.json.

Frontend statis (Next.js/HTML di Vercel) tinggal fetch file JSON ini --
tidak perlu runtime Python di Vercel sama sekali.

CATATAN ARSITEKTUR (lihat juga .github/workflows/daily_update.yml):
Ini BELUM di-deploy. Skrip dan workflow ini disiapkan sesuai permintaan
Tahap 5, tapi keputusan berikut masih perlu dikonfirmasi ke tim sebelum
repo publik dibuat:
    - Apakah dasbor ini versi publik/demo (data 1 hari lag boleh), atau
      perlu near-real-time (arsitektur ini TIDAK cocok untuk itu -- cron
      harian berarti data selalu tertinggal beberapa jam sampai 1 hari).
    - Apakah level inventori "saat ini" diinput manual oleh operator
      (sesuai deskripsi proposal 3.7), atau diasumsikan dari data historis
      (skrip ini memakai volume hari terakhir sebagai proksi, BUKAN input
      manual -- ini adalah simplifikasi yang harus didiskusikan, bukan
      pengganti otomatis untuk "operator memasukkan level inventori saat
      ini" seperti yang dijelaskan proposal).

Usage:
    cd Luaran/scripts/Tahap_5_Prototipe_Pelaporan
    python 03_generate_dashboard_json.py
Writes: data/latest.json
"""

from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Shared modules live one level up in 00_modul_inti/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "00_modul_inti"))

import numpy as np

import config
import monte_carlo
import inventory_policy
import live_data

# Three .parent hops: this file -> Tahap_5_Prototipe_Pelaporan -> scripts -> Luaran
OUT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "latest.json"
LOOKBACK_DAYS = 400  # enough history for a reasonable walk-forward min_history_days

# NOTE: fetch logic now lives in live_data.py (shared with decision_tool_cli.py
# and dashboard_app.py) -- was duplicated here before, consolidated to keep
# the USD-volume unit fix and error handling in exactly one place.


def snapshot_for_asset(asset: str, ticker: str) -> dict:
    df = live_data.fetch_live_asset_data(asset, period_days=LOOKBACK_DAYS)

    # Elasticity from all available history (no separate calibration window
    # in this near-real-time context -- documented simplification vs the
    # proposal's 2022-2024 calibration window, which this dashboard does
    # NOT reproduce; it is a live decision-support tool, not a backtest).
    elasticity = monte_carlo.estimate_price_demand_elasticity(
        df["log_return"].values, df["volume"].values,
    )

    rng = np.random.default_rng(config.RANDOM_SEED)
    out = monte_carlo.simulate_demand_scenarios(
        historical_log_returns=df["log_return"].values,
        historical_volume_mean=df["volume"].mean(),
        elasticity=elasticity,
        lead_time_days=config.LEAD_TIME_DAYS,
        n_scenarios=config.N_SCENARIOS,
        rng=rng,
        exponent_clip=config.EXPONENT_CLIP,
    )
    ss, rop = inventory_policy.compute_ss_rop(out.sigma_d, out.d_bar, out.p95_demand)

    # Proxy "current inventory" from latest observed volume -- see module
    # docstring: this is NOT operator input, it's a placeholder until the
    # dashboard UI collects a real inventory figure.
    current_inventory_proxy = float(df["volume"].tail(1).iloc[0])
    restock_recommended = current_inventory_proxy <= rop

    trend_30d = df.tail(30)[["close", "volume"]].reset_index()
    trend_30d["date"] = trend_30d["date"].dt.strftime("%Y-%m-%d")

    return {
        "asset": asset,
        "ticker": ticker,
        "as_of_date": df.index[-1].strftime("%Y-%m-%d"),
        "elasticity": round(float(elasticity), 4),
        "safety_stock": round(float(ss), 2),
        "reorder_point": round(float(rop), 2),
        "target_inventory_p95": round(float(out.p95_demand), 2),
        "current_inventory_proxy": round(current_inventory_proxy, 2),
        "restock_recommended": bool(restock_recommended),
        "n_exponent_clipped": int(out.n_clipped),
        "trend_30d": trend_30d.to_dict(orient="records"),
    }


def main():
    snapshots = {}
    for asset, ticker in config.YFINANCE_TICKERS.items():
        print(f"Generating snapshot for {asset} ({ticker})...")
        snapshots[asset] = snapshot_for_asset(asset, ticker)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": ("Dasbor demo -- current_inventory_proxy BUKAN input operator riil, "
                 "lihat docstring skrip ini. Data 1 hari lag (cron harian)."),
        "assets": snapshots,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
