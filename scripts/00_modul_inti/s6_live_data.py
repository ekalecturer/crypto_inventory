"""
Helper bersama: ambil data BTC-USD/ETH-USD LANGSUNG dari Yahoo Finance.

Dipakai oleh decision_tool_cli.py dan dashboard_app.py sebagai satu-satunya
sumber data (bukan lagi fallback) -- keduanya dirancang untuk dijalankan di
lingkungan dengan akses internet nyata: mesin operator, GitHub Actions
runner, atau Streamlit Community Cloud (yang deploy langsung dari repo
GitHub). Modul ini TIDAK bergantung pada CSV lokal di data/raw/ sama
sekali.

CATATAN PENTING: ini TIDAK akan berfungsi di sandbox pengembangan yang
memblokir akses keluar ke Yahoo Finance (termasuk sandbox tempat proyek ini
awalnya dikembangkan). Itu wajar -- target deploy (GitHub Actions, Streamlit
Cloud, mesin operator) punya akses internet penuh, sandbox pengembangan
tidak. Kegagalan fetch di sini SELALU dilempar sebagai RuntimeError dengan
pesan jelas, tidak pernah diam-diam jatuh ke data lama atau data kosong.
"""

from __future__ import annotations

import pandas as pd

import s1_config as config
import s2_data_pipeline as data_pipeline


def fetch_live_asset_data(asset: str, period_days: int = 400) -> pd.DataFrame:
    """
    Ambil dan pra-proses `period_days` hari terakhir data harian untuk
    `asset` ("BTC" atau "ETH") langsung dari yfinance.

    Menerapkan perbaikan unit yang sama seperti s2_data_pipeline.py:
    field Volume yfinance untuk ticker kripto adalah dalam USD, bukan
    unit koin -- dikonversi di sini sebelum dikembalikan.

    Raises:
        RuntimeError: jika yfinance tidak terpasang, atau fetch gagal
            (tidak ada internet, ticker tidak valid, dsb). Tidak pernah
            mengembalikan data kosong/lama secara diam-diam.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError(
            "Paket 'yfinance' tidak terpasang. Jalankan: pip install yfinance"
        ) from e

    ticker = config.YFINANCE_TICKERS[asset]
    try:
        raw = yf.download(ticker, period=f"{period_days}d", progress=False, auto_adjust=False)
    except Exception as e:
        raise RuntimeError(f"Gagal mengambil data live untuk {ticker}: {e}") from e

    if raw is None or raw.empty:
        raise RuntimeError(
            f"yfinance mengembalikan data kosong untuk {ticker}. Periksa koneksi "
            f"internet dari lingkungan ini, atau apakah ticker masih valid."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns={"Close": "close", "Volume": "volume"})[["close", "volume"]]
    raw.index.name = "date"
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()

    # UNIT FIX: sama seperti s2_data_pipeline.py -- volume yfinance untuk
    # ticker kripto dalam USD, bukan unit koin.
    raw["volume"] = raw["volume"] / raw["close"]

    df = data_pipeline.preprocess(raw)
    if len(df) < 30:
        # Bukan error fatal, tapi peringatan penting -- elastisitas dan
        # simulasi Monte Carlo tidak stabil dengan riwayat sesingkat ini.
        import sys
        print(f"[warning] hanya {len(df)} hari data live tersedia untuk {ticker} "
              f"setelah pra-pemrosesan -- hasil mungkin tidak stabil.", file=sys.stderr)
    return df
