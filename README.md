# Optimasi Pengisian Ulang Inventori Aset Kripto pada Bursa Terpusat

Simulasi Monte Carlo (Bootstrap) — Pendekatan Safety Stock dan Reorder Point.
Hibah Penelitian LPPM ITK, 2026.

**Hasil (GitHub Pages):** akan aktif di `https://<username>.github.io/<repo>/`
setelah Pages diaktifkan di Settings → Pages (lihat instruksi setup).

**Dasbor interaktif (Streamlit):** akan aktif di
`https://<app-name>.streamlit.app` setelah dideploy di Streamlit Community Cloud.

## Struktur

```
scripts/
├── 00_modul_inti/              modul bersama (config, data_pipeline, monte_carlo,
│                                inventory_policy, backtest, live_data)
├── Tahap_1_Pengumpulan_Data/    fetch data yfinance (dijalankan di luar sandbox)
├── Tahap_2_Model_Simulasi/      validasi KS + fat-tail, grafik distribusi
├── Tahap_3_Turunan_SS_ROP/      grafik parameter SS/ROP harian
├── Tahap_4_Backtesting_Validasi/  backtest proposed vs baseline + grafik
└── Tahap_5_Prototipe_Pelaporan/   alat CLI, dasbor Streamlit, generator JSON

data/            CSV mentah + processed
output/          tabel hasil, grafik, log eksekusi
tests/           unit test (bug fix elastisitas)
index.html       halaman hasil statis (GitHub Pages)
INSTRUCTIONS.md  panduan replikasi lengkap
```

Lihat `INSTRUCTIONS.md` untuk cara menjalankan ulang pipeline, dan `index.html`
untuk hasil backtest yang sudah dikonfirmasi (30 Jul 2026).

## Catatan status

- `.github/workflows/daily_update.yml` ada tapi **belum diaktifkan** — lihat
  komentar di file itu untuk syarat aktivasi.
- Pipeline backtest (Tahap 1-4) memakai CSV lokal di `data/raw/`. Alat Tahap 5
  (CLI + dasbor) memakai data live dari Yahoo Finance secara default.
