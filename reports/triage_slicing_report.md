# Laporan Evaluasi Triase Multi-Task & Slicing Error Analysis

- **Model Mode**: `indobert_lora_trained`
- **NER Exact-Match F1**: `1.0000` (Target >= 0.95)
- **OOF Critical Threshold ($\theta_{crit}^*$)**: `0.495` (Temperature $T = 0.6$)
- **Gap Macro-F1 (Formal vs Slang/Typo)**: `2.34%` (Target <= 8.0%)

## 1. Perbandingan Performa Model pada Held-Out Test Split (180 Tiket)

| Model Konfigurasi | Category Macro-F1 | Urgency Macro-F1 | Critical PR-AUC | Critical Precision | Critical Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline (TF-IDF + LightGBM) | 0.7713 | 0.6259 | 0.6745 | 0.8824 | 0.5357 |
| IndoBERT-LoRA (Default Argmax) | 0.9818 | 0.9644 | 1.0000 | 0.9655 | 1.0000 |
| IndoBERT-LoRA (OOF Threshold Calibrated) | 0.9818 | 0.9644 | 1.0000 | 0.9655 | 1.0000 |

## 2. Slicing Error Analysis Berdasarkan Karakteristik Tiket

| Segmen Sub-Populasi Tiket | Jumlah Sampel | Baseline Macro-F1 | IndoBERT-LoRA Macro-F1 | Delta Peningkatan |
| :--- | :---: | :---: | :---: | :---: |
| Tiket Formal (>15 Kata) | 123 | 0.6912 | 0.9781 | +0.2869 |
| Tiket Pendek (<12 Kata) | 27 | 0.7779 | 0.9500 | +0.1721 |
| Tiket Slang/Typo | 57 | 0.6979 | 0.9547 | +0.2568 |
| Tiket Dengan Kode Error Eksplisit | 115 | 0.7023 | 0.9721 | +0.2698 |
| Tiket Tanpa Kode Error | 65 | 0.6861 | 0.9756 | +0.2895 |
| Tiket Keluhan Multi-Masalah | 12 | 0.5691 | 0.9571 | +0.3880 |

## 3. Daftar 10 Sampel dengan Nilai Loss Tertinggi (Top-Loss Errors)

| Ticket ID | True Category | Pred Category | True Urgency | Pred Urgency | Cross-Entropy Loss | Cuplikan Pesan |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `TCK-00865` | Hardware_POS | Hardware_POS | Medium | Critical | 2.6480 | Yth. Tim Dukungan Operasional Gamatechno, cetak struk pembayaran di mesin kasir macet total disertai bunyi bip peringata... |
| `TCK-00425` | Academic_Portal | Academic_Portal | Medium | High | 1.0891 | Yth. Tim Dukungan Operasional Gamatechno, kegagalan sinkronisasi token otentikasi CAS/OAuth2 pada portal akademik yang m... |
| `TCK-00076` | Hardware_POS | Hardware_POS | Medium | Low | 1.0374 | Yth. Tim Dukungan Operasional Gamatechno, mohon kirim perintah cold-start modul GNSS dan pembaruan A-GPS karena posisi a... |
| `TCK-00523` | Billing_Payment | Billing_Payment | Medium | Low | 0.8954 | Salam sejahtera, bersama tiket ini unit operasional kami melaporkan gangguan: prosedur pembatalan nomor Virtual Account ... |
| `TCK-00423` | Warranty_Claim | Warranty_Claim | Medium | High | 0.8846 | halo gan tolong dibantu segera, pelanggan mengeluhkan servsi penggantian mianboard yang belmu rampung setelah lewat 14 h... |
| `TCK-00631` | Warranty_Claim | Warranty_Claim | Medium | High | 0.8311 | Halo Tim Care Gamatechno, mohon penanganan untuk permasalahan sistem berikut: pelanggan mengeluhkan servis penggantian m... |
| `TCK-00091` | Billing_Payment | Billing_Payment | Medium | Medium | 0.8110 | Selamat siang Layanan Pelanggan & Teknisi Gamatechno, prosedur pembatalan nomor Virtual Account lama dan penyelarasan ul... |
| `TCK-01047` | Warranty_Claim | Warranty_Claim | Medium | High | 0.7966 | halo gan tolong dibantu segera, tiket klaim perbaikan mainboard di portal SIDIG menembus batas 14 hari kerja, bagaimana ... |
| `TCK-00927` | Hardware_POS | Hardware_POS | High | High | 0.6487 | halo gan tolong dibantu segera, unit pelacak kendaraan tidak mengaktifkan fitur telemetri penuh karena terjebak mode dar... |
| `TCK-01145` | Academic_Portal | Academic_Portal | Medium | Medium | 0.6209 | Yth. Tim Dukungan Operasional Gamatechno, gateway otentikasi SSO kampus menolak sesi login mahasiswa dan dosen setelah r... |
