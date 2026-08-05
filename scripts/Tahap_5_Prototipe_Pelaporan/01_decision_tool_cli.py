"""
Tahap 5 -- Alat Bantu Keputusan berbasis Command-Line (Bab 3.7).

Ini BUKAN main.py (yang menjalankan backtest penuh pada data historis
2022-2025 untuk menghasilkan angka resmi Tabel 1 LKP). Ini adalah alat
operasional harian yang dideskripsikan proposal 3.7: operator memasukkan
level inventori BTC/ETH SAAT INI, alat menampilkan SS terkini, ROP terkini,
dan rekomendasi pengisian ulang (ya/tidak).

SUMBER DATA: LIVE dari Yahoo Finance (via scripts/live_data.py), BUKAN CSV
lokal di data/raw/. Perubahan ini disengaja agar alat ini bisa dijalankan
dari repo yang di-deploy/di-clone di mesin mana pun dengan akses internet
(termasuk GitHub Actions runner) tanpa harus menjalankan skrip fetch
terpisah dulu. Versi sebelumnya membaca CSV lokal karena dikembangkan di
sandbox tanpa akses jaringan keluar -- itu tidak lagi jadi kendala begitu
alat ini benar-benar dijalankan di luar sandbox tersebut.

Desain keputusan yang tetap sama (didokumentasikan eksplisit):
  - Elastisitas diestimasi dari SELURUH riwayat yang berhasil diambil
    (default 400 hari terakhir), bukan jendela kalibrasi tetap 2022-2024
    yang dipakai main.py untuk backtest resmi. Angka SS/ROP dari alat ini
    TIDAK selalu sama persis dengan Tabel 1 LKP -- itu memang berbeda
    tujuan (keputusan operasional hari ini vs backtest historis resmi),
    jangan dicampuradukkan.

Usage:
    cd Luaran/scripts/Tahap_5_Prototipe_Pelaporan
    python 01_decision_tool_cli.py --asset BTC --inventory 500000
    python 01_decision_tool_cli.py --asset ETH --inventory 1200000 --asof 2025-06-15
    python 01_decision_tool_cli.py --asset BTC --inventory 500000 --json
    python 01_decision_tool_cli.py --asset BTC --inventory 500000 --local   # lihat catatan di bawah

Flag --local (mode offline/dev SAJA):
    Membaca data/raw/{asset}_yfinance_raw.csv alih-alih fetch live. Hanya
    berguna untuk pengembangan/pengujian di lingkungan tanpa akses
    internet (mis. sandbox Cowork). JANGAN pakai --local untuk keputusan
    restock riil -- datanya bisa sudah usang.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# Shared modules live one level up in 00_modul_inti/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "00_modul_inti"))

import numpy as np

import config
import data_pipeline
import monte_carlo
import inventory_policy
import live_data


def load_data_for_tool(asset: str, local: bool):
    if local:
        print("[warning] --local dipakai: membaca CSV lokal di data/raw/, BUKAN data live. "
              "Hanya untuk pengembangan/pengujian offline -- jangan pakai untuk keputusan "
              "restock riil.", file=sys.stderr)
        try:
            return data_pipeline.load_asset_data(asset)
        except FileNotFoundError as e:
            print(f"[error] {e}", file=sys.stderr)
            sys.exit(1)

    try:
        return live_data.fetch_live_asset_data(asset)
    except RuntimeError as e:
        print(f"[error] {e}", file=sys.stderr)
        print("[info] Jika lingkungan ini memang tidak punya akses internet (mis. sandbox "
              "pengembangan), coba ulangi dengan --local untuk memakai data historis lokal "
              "sebagai gantinya (hanya untuk pengujian, bukan keputusan riil).", file=sys.stderr)
        sys.exit(1)


def compute_recommendation(df, asof: str | None, current_inventory: float) -> dict:
    if asof is not None:
        df = df.loc[df.index <= asof]
        if df.empty:
            raise ValueError(f"Tidak ada data pada atau sebelum {asof}.")

    if len(df) < 30:
        print(f"[warning] Hanya {len(df)} hari data tersedia -- estimasi elastisitas dan "
              f"simulasi Monte Carlo mungkin tidak stabil dengan riwayat sesingkat ini.",
              file=sys.stderr)

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

    restock_needed = current_inventory <= rop
    order_qty = max(0.0, out.p95_demand - current_inventory + ss) if restock_needed else 0.0

    return {
        "as_of_date": df.index[-1].strftime("%Y-%m-%d"),
        "n_history_days_used": int(len(df)),
        "elasticity": round(float(elasticity), 4),
        "current_inventory": round(float(current_inventory), 2),
        "safety_stock": round(float(ss), 2),
        "reorder_point": round(float(rop), 2),
        "target_inventory_p95": round(float(out.p95_demand), 2),
        "restock_recommended": bool(restock_needed),
        "recommended_order_qty": round(float(order_qty), 2),
        "n_exponent_clipped": int(out.n_clipped),
    }


def print_human(asset: str, result: dict, source: str):
    print(f"\n{'='*58}")
    print(f"  ALAT BANTU KEPUTUSAN INVENTORI -- {asset}  [{source}]")
    print(f"  Per tanggal data: {result['as_of_date']}  "
          f"({result['n_history_days_used']} hari riwayat dipakai)")
    print(f"{'='*58}")
    print(f"  Inventori saat ini      : {result['current_inventory']:,.2f}")
    print(f"  Safety Stock (SS)       : {result['safety_stock']:,.2f}")
    print(f"  Reorder Point (ROP)     : {result['reorder_point']:,.2f}")
    print(f"  Target Inventori (P95)  : {result['target_inventory_p95']:,.2f}")
    print(f"  Elastisitas (beta)      : {result['elasticity']:.4f}")
    if result["n_exponent_clipped"] > 0:
        print(f"  [PERINGATAN] exponent clip kena {result['n_exponent_clipped']}x -- "
              f"lihat monte_carlo.py")
    print(f"{'-'*58}")
    if result["restock_recommended"]:
        print(f"  REKOMENDASI: >>> LAKUKAN PENGISIAN ULANG SEKARANG <<<")
        print(f"  Jumlah order yang disarankan: {result['recommended_order_qty']:,.2f}")
        print(f"  Alasan: inventori saat ini ({result['current_inventory']:,.2f}) "
              f"<= ROP ({result['reorder_point']:,.2f})")
    else:
        print(f"  REKOMENDASI: TAHAN -- belum perlu pengisian ulang")
        print(f"  Alasan: inventori saat ini ({result['current_inventory']:,.2f}) "
              f"> ROP ({result['reorder_point']:,.2f})")
    print(f"{'='*58}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Alat bantu keputusan restock inventori kripto (Bab 3.7 proposal). "
                     "Default: data LIVE dari Yahoo Finance."
    )
    parser.add_argument("--asset", required=True, choices=config.ASSETS)
    parser.add_argument("--inventory", required=True, type=float,
                         help="Level inventori saat ini (unit aset)")
    parser.add_argument("--asof", default=None,
                         help="Hitung seolah-olah hari ini adalah tanggal ini (YYYY-MM-DD). "
                              "Default: tanggal terakhir yang tersedia di data live.")
    parser.add_argument("--local", action="store_true",
                         help="Mode offline/dev: baca CSV lokal di data/raw/ alih-alih fetch "
                              "live. Jangan pakai untuk keputusan restock riil.")
    parser.add_argument("--json", action="store_true", help="Cetak output sebagai JSON.")
    args = parser.parse_args()

    df = load_data_for_tool(args.asset, args.local)
    result = compute_recommendation(df, args.asof, args.inventory)
    source = "data lokal (--local)" if args.local else "data live (yfinance)"

    if args.json:
        print(json.dumps({"asset": args.asset, "source": source, **result}, indent=2))
    else:
        print_human(args.asset, result, source)


if __name__ == "__main__":
    main()
