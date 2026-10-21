# Optimasi Pengisian Ulang Inventori Aset Kripto pada Bursa Terpusat

Simulasi Monte Carlo (Bootstrap) — Pendekatan Safety Stock dan Reorder Point.
Hibah Penelitian LPPM ITK, 2026.

**Hasil (GitHub Pages):** https://ekalecturer.github.io/crypto_inventory/

**Dasbor interaktif (Streamlit):** belum di-deploy sebagai layanan publik;
jalankan lokal dengan `streamlit run scripts/Tahap_5_Prototipe_Pelaporan/02_dashboard_app.py`.

## Struktur

```
scripts/
├── 00_modul_inti/              modul bersama (config, data_pipeline, monte_carlo,
│                                inventory_policy, backtest, live_data)
├── Tahap_1_Pengumpulan_Data/    fetch data yfinance (dijalankan di luar sandbox)
├── Tahap_2_Model_Simulasi/      validasi KS + fat-tail, grafik distribusi
├── Tahap_3_Turunan_SS_ROP/      grafik parameter SS/ROP harian
├── Tahap_4_Backtesting_Validasi/  backtest proposed vs baseline + grafik
├── Tahap_5_Prototipe_Pelaporan/   alat CLI, dasbor Streamlit, generator JSON
└── Tahap_6_Robustness_Hourly/     uji ketahanan pada data per jam (di luar proposal)

data/            CSV mentah + processed
output/          tabel hasil, grafik, log eksekusi (resolusi harian)
output_hourly/   tabel dan grafik Tahap 6 (resolusi per jam)
tests/           unit test (bug fix elastisitas)
index.html       halaman hasil statis (GitHub Pages)
INSTRUCTIONS.md  panduan replikasi lengkap
```

Lihat `INSTRUCTIONS.md` untuk cara menjalankan ulang pipeline, dan `index.html`
untuk hasil backtest yang sudah dikonfirmasi (30 Jul 2026).

## Tahap 6: uji ketahanan data per jam

Uji ketahanan tambahan **di luar proposal asli**. Hasil utama penelitian tetap
backtest harian (Tahap 4). Tahap 6 mengulang metodologi yang sama pada candlestick
1 jam untuk melihat apakah kesimpulan berubah bila resolusi data diperhalus.

Parameter: lead time L = 24 periode jam (1 hari kalender), baseline rolling 720 jam,
biaya holding per jam = biaya harian / 24, ambang gap 72 jam, N = 10.000 skenario
per periode, kalibrasi 2022–2024, backtest penuh 2025 tanpa subsampel (8.657 periode
BTC, 8.609 periode ETH). Elastisitas diestimasi ulang dari data jam:
β BTC = -2,1513, β ETH = -3,0304.

| Aset | Stockout proposed | Stockout baseline | Holding/jam proposed | Holding/jam baseline | p |
|---|---|---|---|---|---|
| BTC | 0,00% | 27,25% | 71.080 | 296 | < 0,001 |
| ETH | 0,01% | 27,92% | 12.660 | 235 | < 0,001 |

Pembanding harian (Tahap 4): stockout BTC 8,49% vs 33,70%, ETH 61,92% vs 32,05%.

Kesimpulan kualitatif sama dengan hasil harian: kriteria superioritas (stockout
**dan** biaya holding sama-sama lebih rendah secara signifikan) tidak terpenuhi pada
kedua aset. Pola trade-off-nya berubah: stockout mendekati nol dengan biaya holding
jauh lebih tinggi, sehingga hasil sensitif terhadap resolusi data.

Hasil ada di `output_hourly/tables/` (`hourly_metrics_summary.csv`,
`hourly_ks_validation.csv`, hasil per periode) dan `output_hourly/figures/`.

**Data per jam tidak disertakan di repo.** Sumbernya CryptoDataDownload (Binance,
1 jam) dengan lisensi CC BY-NC-SA 4.0. Untuk menjalankan ulang, unduh ke
`data/raw/`:

- https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_1h.csv
- https://www.cryptodatadownload.com/cdd/Binance_ETHUSDT_1h.csv

lalu jalankan `python scripts/Tahap_6_Robustness_Hourly/01_run_hourly_robustness.py`
(perlu waktu lama; menimpa isi `output_hourly/`). Penyedia memperbarui berkas ini
terus-menerus, sehingga unduhan baru bisa berbeda sedikit dari salinan yang dipakai
untuk hasil di atas (salinan lokal bertanggal 12 Agustus 2026).

## Catatan status

- `.github/workflows/daily_update.yml` menulis `data/latest.json`. Jadwal
  hariannya dimatikan sejak 21 Okt 2026 (hanya bisa dijalankan manual) karena
  belum ada komponen yang membaca berkas itu; dasbor Streamlit mengambil data
  langsung dari Yahoo Finance.
- Pipeline backtest (Tahap 1-4) memakai CSV lokal di `data/raw/`. Alat Tahap 5
  (CLI + dasbor) memakai data live dari Yahoo Finance secara default.
