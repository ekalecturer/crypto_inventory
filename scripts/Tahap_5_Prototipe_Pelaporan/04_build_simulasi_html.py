"""
Membangun simulasi/index.html (simulasi Monte Carlo yang berjalan di browser).

Data harga dan volume dibaca lewat modul inti yang sama dengan pipeline
backtest (volume sudah dikonversi dari USD ke unit aset), lalu ditanam ke
halaman sebagai array JSON. Seluruh simulasi berjalan di sisi klien, jadi
halaman dapat di-host di GitHub Pages tanpa server.

Pemakaian:
    cd scripts/Tahap_5_Prototipe_Pelaporan
    python 04_build_simulasi_html.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "00_modul_inti"))
import s2_data_pipeline as dp  # noqa: E402

ROOT = HERE.parent.parent
TEMPLATE = HERE / "simulasi_template.html"
OUT = ROOT / "simulasi" / "index.html"

data = {}
for asset in ("BTC", "ETH"):
    df = dp.load_asset_data(asset)
    data[asset] = {
        "start": df.index[0].strftime("%Y-%m-%d"),
        "c": [round(float(x), 2) for x in df["close"]],
        "v": [round(float(x), 2) for x in df["volume"]],
        "r": [round(float(x), 6) for x in df["log_return"]],
    }

html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", json.dumps(data, separators=(",", ":")))
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(html, encoding="utf-8")
print(f"Ditulis {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
