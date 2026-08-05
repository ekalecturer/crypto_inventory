# Instruksi Replikasi Pipeline — Optimasi Inventori Kripto (Bootstrap-MC / SS-ROP)

> **STATUS: Fase 0 SELESAI dan angka sudah dikonfirmasi Eka (30 Jul 2026).**
> Hasil final ada di `output/tables/metrics_summary.csv` dan
> `output/logs/full_run_ALL.log`. Jangan jalankan ulang pipeline dengan
> mengubah konfigurasi apa pun tanpa alasan yang dicatat — ini bukan lagi
> tahap uji coba.

Dokumen ini untuk Laode/Siti (atau siapa pun yang menjalankan ulang pipeline tanpa
supervisi langsung). Ikuti urutan ini persis. Jangan lompat langkah, jangan
mengganti angka konfigurasi tanpa mencatat alasannya di log.

Ini adalah pipeline untuk hibah LPPM "Optimasi Pengisian Ulang Inventori Aset
Kripto ... Bootstrap-MC / Safety Stock / Reorder Point". **Bukan** pipeline
GBM/VaR/Kupiec (itu proyek terpisah — JAMB/BICAME — folder lain, jangan dicampur).

## 0. Latar belakang wajib dibaca dulu

Baca `Draft_Artikel_Preliminer_Monte_Carlo_Inventori_Kripto_rev.docx` (satu
folder di atas ini) — Bagian 5 (Keterbatasan) khususnya. Itu mendokumentasikan
dua masalah yang pipeline ini memperbaiki:

1. **BTC**: algoritma dinamis mencatat biaya holding signifikan lebih tinggi
   dari baseline statis (p<0,0001) pada uji coba awal — bukan bug, ini temuan
   riil yang harus dilaporkan apa adanya, bukan "diperbaiki" sampai hilang.
2. **ETH**: bug numerik pada fungsi elastisitas permintaan (`exp(elasticity * r)`
   tanpa batas) meledak saat window walk-forward menyerap return ekstrem era
   krisis (mis. keruntuhan FTX, ~-55% dalam satu hari). **Ini yang sudah
   diperbaiki** di `scripts/00_modul_inti/s3_monte_carlo.py` (lihat docstring "BUG FIX"
   di file itu) — exponent-nya sekarang dibatasi (clipped) ke ±10 sebelum di-exponensial-kan.

**Dua bug tambahan ditemukan saat menjalankan pipeline ini pada data riil
(30 Jul 2026), keduanya sudah diperbaiki sebelum angka final dihasilkan:**

3. **Unit bug pada volume**: field `Volume` yfinance untuk BTC-USD/ETH-USD
   ternyata dalam USD (dolar), bukan jumlah koin — buktinya: volume BTC di
   2022-01-01 tercatat ~24,6 miliar, padahal total suplai BTC hanya ~19 juta
   koin. Tanpa dikonversi, semua angka inventori/demand/holding cost di hilir
   akan salah dengan faktor sebesar harga aset. **Sudah diperbaiki** di
   `scripts/00_modul_inti/s2_data_pipeline.py`: `volume = volume_usd / close`.
4. **Bug overwrite pada export**: menjalankan `--asset BTC` lalu `--asset ETH`
   terpisah (seperti instruksi langkah 5 versi lama) membuat baris BTC di
   `metrics_summary.csv` tertimpa hilang oleh run ETH. **Sudah diperbaiki**
   di `scripts/Tahap_4_Backtesting_Validasi/01_main_backtest.py` — `export_results()`
   sekarang menggabungkan (merge),
   bukan menimpa. Tetap disarankan pakai `--asset ALL` sekali jalan (lihat
   langkah 5 di bawah, sudah diperbarui).

## 1. Instalasi environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install pandas numpy scipy yfinance pytest
```

## 2. Struktur folder

> **Diperbarui setelah reorganisasi proyek (5 Agu 2026):** semua isi teknis
> (scripts, data, output, tests, .github) sekarang bersarang di dalam
> `Luaran/` — folder ini akan menjadi root repo GitHub saat di-deploy.
> `Proposal/`, `Laporan Kemajuan 70%/`, dan `Laporan Kemajuan 100%/` berada
> di level yang sama dengan `Luaran/`, berisi dokumen administratif, bukan
> kode. Template LPPM tetap di root, tidak dipindah ke folder manapun.

```
inventori_hibah_pipeline/
├── Proposal/                          <- dokumen proposal + Gambar 1/2 + Diagram 3.1
├── Laporan Kemajuan 70%/              <- logbook checkpoint 70%
├── Laporan Kemajuan 100%/             <- LKP final, slide pendamping, checklist
├── template laporan kemajuan/         <- template LPPM, TIDAK dipindah
├── template hak cipta/                <- template LPPM, TIDAK dipindah
└── Luaran/                            <- root repo GitHub (kode + hasil)
    ├── INSTRUCTIONS.md                <- dokumen ini
    ├── requirements.txt
    ├── data/
    │   ├── raw/                       <- CSV mentah dari yfinance (lihat langkah 3)
    │   └── processed/                 <- (tidak dipakai otomatis; processed data in-memory)
    ├── scripts/
    │   ├── 00_modul_inti/             <- modul bersama, dipakai semua Tahap
    │   │   ├── s1_config.py           <- konstanta, dengan komentar sumber (proposal vs asumsi)
    │   │   ├── s2_data_pipeline.py    <- load CSV lokal + preprocessing (gap, log-return)
    │   │   ├── s3_monte_carlo.py      <- bootstrap MC + elasticity scaling (SUDAH DIPERBAIKI)
    │   │   ├── s4_inventory_policy.py <- SS/ROP dinamis + baseline fixed buffer
    │   │   ├── s5_backtest.py         <- metrik + paired t-test
    │   │   └── s6_live_data.py        <- fetch live yfinance (dipakai Tahap 5)
    │   │       (penomoran s1-s6 = urutan pakai; prefiks huruf karena modul
    │   │        Python tidak bisa diawali angka dan diimpor langsung)
    │   ├── Tahap_1_Pengumpulan_Data/
    │   │   └── 01_fetch_yfinance_LOCAL.py   <- HARUS dijalankan di luar sandbox, ada internet
    │   ├── Tahap_2_Model_Simulasi/
    │   │   ├── 01_ks_validation.py
    │   │   └── 02_plot_fat_tail_distribution.py
    │   ├── Tahap_3_Turunan_SS_ROP/
    │   │   └── 01_plot_ssrop_timeseries.py
    │   ├── Tahap_4_Backtesting_Validasi/
    │   │   ├── 01_main_backtest.py    <- orkestrasi end-to-end, CLI entry point (dulu main.py)
    │   │   └── 02_plot_backtest_results.py
    │   └── Tahap_5_Prototipe_Pelaporan/
    │       ├── 01_decision_tool_cli.py
    │       ├── 02_dashboard_app.py
    │       └── 03_generate_dashboard_json.py
    ├── tests/
    │   └── test_monte_carlo_elasticity_fix.py   <- WAJIB lulus sebelum lanjut ke data riil
    ├── output/
    │   ├── tables/                    <- hasil CSV (metrics_summary.csv, daily series)
    │   ├── figures/                   <- grafik PNG Tahap 2-4
    │   └── logs/                      <- simpan stdout setiap run di sini (lihat langkah 6)
    └── .github/workflows/daily_update.yml   <- belum diaktifkan, lihat catatan di file itu
```

## 3. Ambil data (WAJIB dijalankan DI LUAR sandbox Cowork)

Sandbox Cowork tidak bisa menjangkau Yahoo Finance atau CoinGecko (diblokir di
level proxy jaringan). Jalankan langkah ini di laptop Anda sendiri atau di
Google Colab:

```bash
cd Luaran/scripts/Tahap_1_Pengumpulan_Data
python 01_fetch_yfinance_LOCAL.py
```

Ini akan:
- Mengunduh BTC-USD dan ETH-USD harian, 1 Jan 2022 – 31 Des 2025
- Mencetak laporan gap (hari yang hilang, dan apakah akan diinterpolasi
  atau di-drop sesuai aturan Bab 3.3: gap <3 hari = interpolasi linear,
  gap ≥3 hari = drop)
- Menulis `data/raw/BTC_yfinance_raw.csv` dan `data/raw/ETH_yfinance_raw.csv`

**Salin output terminal (laporan gap) dan simpan** — itu bukti dokumentasi
gap yang wajib dilampirkan ke Laporan Kemajuan (Fase 0 aturan 2 di prompt awal).

Setelah selesai, upload kedua file CSV itu ke folder `Luaran/data/raw/` di sini.

## 4. Jalankan unit test elastisitas SEBELUM data riil

```bash
cd Luaran/tests
python test_monte_carlo_elasticity_fix.py
```

Wajib semua 6 test lulus (`6/6 passed`). Test ini membuktikan bug exp()
yang lama benar-benar overflow pada input ekstrem, dan versi yang di-clip
tetap finite pada input yang sama. Kalau ada test gagal, JANGAN lanjut ke
langkah 5 — laporkan kegagalannya dulu.

## 5. Jalankan pipeline penuh

```bash
cd Luaran/scripts/Tahap_4_Backtesting_Validasi
python 01_main_backtest.py --asset ALL 2>&1 | tee ../../../output/logs/full_run_ALL.log
```

Jalankan `--asset ALL` sekali jalan (bukan BTC lalu ETH terpisah) — walaupun
bug overwrite di export sudah diperbaiki (lihat bug #4 di atas), satu run
`ALL` tetap paling sederhana dan paling kecil risikonya untuk log yang bersih.

Setiap run mencetak:
- Estimasi elastisitas (beta) dari calibration window
- Peringatan jika exponent clip pernah kena (`n_clipped > 0`) — **ini WAJIB
  dilaporkan apa adanya**, bukan disembunyikan, karena berarti clip benar-benar
  mempengaruhi hasil, bukan sekadar jaring pengaman teoretis
- Tabel perbandingan proposed vs baseline (stockout rate, holding cost,
  restock frequency, total stockout)
- Hasil paired t-test per metrik
- Verdict: apakah kriteria superioritas proposal terpenuhi (stockout DAN
  holding cost signifikan lebih rendah, keduanya, alpha=0.05)

Output CSV ditulis ke `output/tables/`:
- `metrics_summary.csv` — satu baris per metrik per aset, format panjang
- `{ASET}_proposed_daily.csv` / `{ASET}_baseline_daily.csv` — seri harian penuh

## 6. Cara membaca output

Buka `output/tables/metrics_summary.csv`. Kolom penting:
- `proposed` vs `baseline`: nilai metrik masing-masing kebijakan
- `p_value`, `significant`: hasil paired t-test
- `proposed_better`: True hanya jika signifikan DAN arah perbedaannya
  menguntungkan proposed (lebih rendah untuk stockout/holding cost)
- `elasticity`: beta yang diestimasi dari calibration window aset itu
- `n_exponent_clipped`: berapa kali clip exponent kena selama backtest —
  kalau ini besar, tulis di laporan bahwa hasil sensitif terhadap pilihan
  batas clip (config.EXPONENT_CLIP = 10.0), bukan angka yang datang murni
  dari data.

**Jangan menyalin angka dari `output/_smoke_test_do_not_use/` ke laporan
apa pun** — folder itu berisi hasil dari data sintetis (uji asap kode saja),
bukan data riil.

## 6b. Hasil final terkonfirmasi (30 Jul 2026)

Sumber: `output/tables/metrics_summary.csv`, log lengkap di
`output/logs/full_run_ALL.log`. Data: 1.461 hari kalender, 2022-01-01 s/d
2025-12-31, tanpa gap (kripto berdagang 24/7, tidak ada hari libur bursa).
Elastisitas hasil regresi log-log pada calibration window: BTC beta =
-1.0552, ETH beta = -1.0839. Exponent clip (±10) tidak pernah kena pada
data riil untuk kedua aset (`n_exponent_clipped = 0`) — fix-nya terbukti
aman lewat unit test, dan pada run nyata ini tidak membatasi apa pun.

| Aset | Stockout rate (proposed vs baseline) | Holding cost (proposed vs baseline) | Kriteria superioritas proposal |
|---|---|---|---|
| BTC | 8,49% vs 33,70% — **signifikan lebih baik** (p<0,001) | 6.740.510 vs 2.339.719 — **signifikan lebih buruk** (p<0,001) | TIDAK terpenuhi (perlu keduanya lebih baik) |
| ETH | 61,92% vs 32,05% — **signifikan lebih buruk** (p<0,001) | 427.342 vs 1.255.350 — **signifikan lebih baik** (p<0,001) | TIDAK terpenuhi |

Pola trade-off-nya berlawanan arah di kedua aset: BTC menang di proteksi
stockout tapi kalah di biaya holding; ETH sebaliknya. Restock frequency
nyaris identik di kedua kebijakan untuk kedua aset (bukan bagian dari
kriteria superioritas proposal). Catatan interpretasi: `total_stockout_amount`
dan metrik absolut lainnya memakai volume gabungan multi-bursa (proxy
demand, bukan order book satu CEX riil) — jangan dikutip sebagai ukuran
reserve exchange tertentu; yang bermakna secara statistik adalah
perbandingan proposed-vs-baseline, bukan angka absolutnya.

## 7. Aturan mutlak (sama seperti proyek JAMB di folder sebelah)

1. Tidak ada angka yang dilaporkan tanpa log eksekusi (`output/logs/*.log`).
2. Jika komputasi gagal, tulis "TIDAK DAPAT DIHITUNG" + alasannya — jangan
   diisi dengan estimasi atau angka dari draft preliminer lama.
3. Jika hasil BTC masih menunjukkan biaya holding lebih tinggi dari baseline
   (seperti temuan preliminer), itu BUKAN kesalahan yang harus "diperbaiki
   sampai hilang" — itu adalah temuan yang harus dilaporkan jujur sebagai
   trade-off, sesuai keputusan framing di prompt awal proyek ini.
4. Sebelum angka apa pun dari langkah 5 dipakai untuk menulis Laporan
   Kemajuan, artikel, atau dokumen lain — tunjukkan dulu ke Eka untuk
   konfirmasi (lihat Fase 0 aturan 5 di prompt awal).
