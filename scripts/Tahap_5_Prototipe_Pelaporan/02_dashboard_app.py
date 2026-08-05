"""
Tahap 5 -- Dasbor Interaktif (Bab 3.7).

Implementasi Streamlit dari deskripsi proposal: "Operator memasukkan level
inventori BTC atau ETH saat ini, dan dasbor menampilkan nilai SS terkini,
nilai ROP terkini, rekomendasi pengisian ulang (lakukan pengisian sekarang
atau tahan), serta grafik tren level inventori terhadap SS dan ROP selama
30 hari terakhir."

SUMBER DATA: LIVE dari Yahoo Finance (via scripts/live_data.py), BUKAN CSV
lokal di data/raw/. Diubah agar dasbor ini bisa langsung di-deploy dari
repo GitHub (mis. Streamlit Community Cloud, yang menjalankan app ini di
server dengan akses internet penuh) tanpa perlu menjalankan skrip fetch
terpisah dulu atau meng-commit data mentah ke repo.

Fetch di-cache dengan TTL 30 menit (st.cache_data(ttl=1800)) supaya
interaksi slider/tombol operator tidak memicu fetch baru ke Yahoo Finance
setiap kali -- data pasar harian tidak berubah cukup cepat untuk butuh
refresh lebih sering dari itu.

CATATAN JUJUR SOAL GRAFIK TREN 30 HARI
=======================================
Proposal meminta "tren level inventori" -- tapi level inventori riil suatu
CEX adalah data internal operator yang TIDAK tersedia dari data pasar
publik (harga/volume). Dasbor ini TIDAK mengarang data inventori historis.
Yang ditampilkan adalah tren SS_t dan ROP_t (dihitung walk-forward dari
data pasar riil), dengan input inventori operator hari ini diplot sebagai
titik referensi.

Deploy (Streamlit Community Cloud):
    1. Push repo ini ke GitHub.
    2. share.streamlit.io -> New app -> pilih repo, branch, dan file ini
       (scripts/Tahap_5_Prototipe_Pelaporan/02_dashboard_app.py) sebagai
       entry point.
    3. Pastikan requirements.txt di root repo terdeploy bersamanya.

Jalankan lokal (butuh akses internet ke Yahoo Finance):
    cd Luaran/scripts/Tahap_5_Prototipe_Pelaporan
    streamlit run 02_dashboard_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Shared modules live one level up in 00_modul_inti/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "00_modul_inti"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import monte_carlo
import inventory_policy
import live_data

st.set_page_config(page_title="Dasbor Inventori Kripto -- SS/ROP", layout="wide")


@st.cache_data(show_spinner="Mengambil data live dari Yahoo Finance...", ttl=1800)
def load_asset_live(asset: str) -> pd.DataFrame:
    return live_data.fetch_live_asset_data(asset)


@st.cache_data(show_spinner=False, ttl=1800)
def compute_trailing_ssrop(asset: str, trailing_days: int = 30) -> pd.DataFrame:
    """SS/ROP walk-forward untuk N hari perdagangan terakhir yang tersedia."""
    df = load_asset_live(asset)
    elasticity = monte_carlo.estimate_price_demand_elasticity(
        df["log_return"].values, df["volume"].values,
    )
    tail_dates = df.index[-trailing_days:]
    rng = np.random.default_rng(config.RANDOM_SEED)
    rows = []
    for date in tail_dates:
        hist = df.loc[:date]
        out = monte_carlo.simulate_demand_scenarios(
            historical_log_returns=hist["log_return"].values,
            historical_volume_mean=hist["volume"].mean(),
            elasticity=elasticity,
            lead_time_days=config.LEAD_TIME_DAYS,
            n_scenarios=2000,  # dikurangi dari 10.000 agar dasbor responsif interaktif
            rng=rng,
            exponent_clip=config.EXPONENT_CLIP,
        )
        ss, rop = inventory_policy.compute_ss_rop(out.sigma_d, out.d_bar, out.p95_demand)
        rows.append({"date": date, "ss": ss, "rop": rop, "target_p95": out.p95_demand})
    result = pd.DataFrame(rows).set_index("date")
    result.attrs["elasticity"] = elasticity
    return result


st.title("Dasbor Keputusan Inventori Kripto -- Safety Stock / Reorder Point")
st.caption(
    "Prototipe Tahap 5 (Bab 3.7 proposal). Data pasar LIVE dari Yahoo Finance. "
    "Level inventori adalah input manual operator, bukan data pasar."
)

with st.sidebar:
    st.header("Input Operator")
    asset = st.selectbox("Aset", config.ASSETS)
    current_inventory = st.number_input(
        "Level inventori saat ini (unit aset)", min_value=0.0, value=100000.0, step=1000.0,
    )
    trailing_days = st.slider("Rentang tren (hari)", 10, 60, 30)
    compute_btn = st.button("Hitung Rekomendasi", type="primary")

if compute_btn or "last_result" in st.session_state:
    try:
        with st.spinner("Menjalankan simulasi Monte Carlo..."):
            trend = compute_trailing_ssrop(asset, trailing_days)
            elasticity = trend.attrs["elasticity"]
            today = trend.iloc[-1]
            st.session_state["last_result"] = (asset, current_inventory, trend, elasticity, today)
    except RuntimeError as e:
        st.error(
            f"**Gagal mengambil data live dari Yahoo Finance:** {e}\n\n"
            f"Jika Anda menjalankan ini di lingkungan tanpa akses internet keluar "
            f"(mis. sandbox pengembangan), ini yang diharapkan -- dasbor ini dirancang "
            f"untuk deploy di lingkungan dengan akses internet nyata (Streamlit Community "
            f"Cloud, mesin operator, dsb)."
        )
        st.stop()

    asset, current_inventory, trend, elasticity, today = st.session_state["last_result"]
    restock_needed = current_inventory <= today["rop"]
    order_qty = max(0.0, today["target_p95"] - current_inventory + today["ss"]) if restock_needed else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Safety Stock (SS)", f"{today['ss']:,.0f}")
    col2.metric("Reorder Point (ROP)", f"{today['rop']:,.0f}")
    col3.metric("Target Inventori (P95)", f"{today['target_p95']:,.0f}")
    col4.metric("Inventori Saat Ini", f"{current_inventory:,.0f}")

    if restock_needed:
        st.error(
            f"**REKOMENDASI: LAKUKAN PENGISIAN ULANG SEKARANG** -- "
            f"inventori ({current_inventory:,.0f}) <= ROP ({today['rop']:,.0f}). "
            f"Jumlah order yang disarankan: **{order_qty:,.0f}**"
        )
    else:
        st.success(
            f"**REKOMENDASI: TAHAN** -- inventori ({current_inventory:,.0f}) "
            f"> ROP ({today['rop']:,.0f}). Belum perlu pengisian ulang."
        )

    st.caption(f"Elastisitas harga-permintaan (beta) diestimasi dari data live: "
               f"{elasticity:.4f}. Per tanggal data: {trend.index[-1].date()} "
               f"(cache diperbarui setiap 30 menit).")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend.index, y=trend["rop"], name="ROP", line=dict(color="#F2A65A", dash="dash")))
    fig.add_trace(go.Scatter(x=trend.index, y=trend["ss"], name="Safety Stock", line=dict(color="#5B6472", dash="dot")))
    fig.add_trace(go.Scatter(
        x=[trend.index[-1]], y=[current_inventory], name="Inventori (input operator)",
        mode="markers", marker=dict(color="#1E2761", size=14, symbol="diamond"),
    ))
    fig.update_layout(
        title=f"{asset}: Tren SS/ROP {trailing_days} Hari Terakhir vs Inventori Saat Ini",
        xaxis_title="Tanggal", yaxis_title="Unit aset",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Catatan metodologi dan keterbatasan"):
        st.markdown(
            "- Data pasar diambil **live** dari Yahoo Finance saat aplikasi dijalankan "
            "(di-cache 30 menit) -- bukan dari CSV lokal.\n"
            "- Elastisitas di dasbor ini diestimasi dari **riwayat live yang tersedia** "
            "(default 400 hari terakhir), berbeda dari jendela kalibrasi tetap 2022-2024 "
            "yang dipakai pada backtest resmi (Tabel 1 LKP) -- angka SS/ROP di sini TIDAK "
            "harus identik dengan angka LKP.\n"
            "- Garis tren yang ditampilkan adalah SS/ROP historis, **bukan** level inventori "
            "historis riil (data itu tidak tersedia dari sumber pasar publik).\n"
            "- N skenario Monte Carlo dikurangi menjadi 2.000 (dari 10.000) untuk performa "
            "interaktif -- hasil bisa sedikit berbeda dari angka resmi yang memakai N=10.000."
        )
else:
    st.info("Masukkan level inventori di sidebar, lalu klik **Hitung Rekomendasi**.")
