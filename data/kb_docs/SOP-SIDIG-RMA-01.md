# SOP-SIDIG-RMA-01: Verifikasi Garansi Nomor Seri, Klaim Penggantian Unit (RMA-08), dan Penolakan Garansi Cairan/Segel Rusak (RMA-12)
Product Model: SIDIG-WARRANTY
Category: Warranty_Claim

## 1. Verifikasi Masa Garansi Aktif & Alur Klaim Penggantian Unit Baru (RMA-08)
### Gejala & Indikasi Teknis RMA-08
Pelanggan mengajukan permohonan klaim purnajual melalui platform **SIDIG After-Sales** dengan kode tiket klaim `RMA-08` (Standard Defective Unit Replacement Claim) untuk perangkat keras yang mengalami cacat produksi pabrik dalam masa garansi resmi 12 bulan. Pelanggan menyertakan nomor seri perangkat berformat `SN-xxxxx`.

### Langkah Perbaikan & Prosedur Klaim RMA-08
1. Buka dashboard **SIDIG Warranty Portal > Verifikasi Nomor Seri**, lalu masukkan kode `SN-xxxxx` untuk memeriksa tanggal aktivasi faktur pembelian dan status garansi (`ACTIVE_WARRANTY`).
2. Pastikan pelanggan telah mengunggah tiga dokumen wajib: foto label nomor seri pada bodi perangkat, salinan faktur pembelian resmi, dan video singkat yang menunjukkan gejala kerusakan perangkat.
3. Apabila perangkat mengalami mati total atau cacat komponen utama di bawah 30 hari sejak pembelian (Dead on Arrival / DOA), terbitkan surat persetujuan **RMA-08 Replacement Approval** untuk pengiriman unit pengganti.
4. Berikan nomor resi pengiriman balik ke gudang pusat Gamatechno dan perbarui status pelacakan di portal SIDIG.

### Batas Eskalasi Teknis RMA-08
Klaim penggantian unit penuh tanpa pemeriksaan fisik laboratorium hanya diizinkan jika masa pakai perangkat kurang dari 14 hari kalender dan segel hologram garansi utuh 100%.

## 2. Kriteria Penolakan Klaim Garansi Akibat Cairan, Terbakar, atau Segel Rusak (RMA-12)
### Gejala & Indikasi Teknis RMA-12
Hasil pemeriksaan awal atau laporan keluhan pelanggan menunjukkan perangkat terkena tumpahan air/minuman, terendam banjir, komponen gosong terbakar akibat lonjakan listrik eksternal, atau stiker segel garansi hologram telah robek/dibongkar pihak tidak resmi. Sistem SIDIG menandai klaim dengan kode `RMA-12` (Warranty Void — Liquid Damage / Tampered Seal).

### Aturan Bisnis & Penanganan Klaim RMA-12
1. Berdasarkan **Kebijakan Garansi Purnajual SIDIG Pasal 4 Ayat 2**, kerusakan akibat masuknya cairan (indikator kertas lakmus internal berubah merah), bencana alam, modifikasi tidak sah, atau segel garansi rusak menyebabkan **garansi gugur secara permanen (VOID)**.
2. **DILARANG KERAS** menyetujui penggantian unit gratis pada kasus `RMA-12`. Agen layanan pelanggan tidak boleh menjanjikan perbaikan cuma-cuma.
3. Sampaikan secara sopan kepada pelanggan bahwa status klaim adalah **Out-of-Warranty Paid Repair (Perbaikan Berbayar)** dan arahkan tiket ke tim estimasi biaya servis resmi untuk penerbitan kuotasi suku cadang.

### Batas Eskalasi Teknis RMA-12
Jika pelanggan korporat menolak hasil temuan indikator cairan `RMA-12` dan meminta peninjauan ulang, eskalasikan tiket ke Manajer Layanan Purnajual SIDIG beserta bukti foto makro papan sirkuit (PCB).
