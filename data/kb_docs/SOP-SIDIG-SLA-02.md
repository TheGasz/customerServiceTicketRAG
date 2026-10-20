# SOP-SIDIG-SLA-02: Prosedur Peminjaman Unit Cadangan Korporat (SLA-GOLD) dan Eskalasi Penggantian Mainboard (RMA-19)
Product Model: SIDIG-WARRANTY
Category: Warranty_Claim

## 1. Prosedur Peminjaman Unit Cadangan (Loaner Unit) Klien Korporat (SLA-GOLD)
### Gejala & Indikasi Teknis SLA-GOLD
Klien korporat pemegang kontrak perlindungan **SIDIG Enterprise SLA-GOLD** melaporkan gangguan perangkat operasional utama dan membutuhkan unit pinjaman sementara (`Loaner Unit`) agar layanan bisnis mereka tidak terhenti selama perangkat asli diperbaiki di pusat servis.

### Langkah Layanan & Prosedur SLA-GOLD
1. Validasi nomor kontrak korporat dan nomor seri perangkat `SN-xxxxx` pada modul **SIDIG Enterprise Contract Registry** untuk memastikan tier layanan adalah `SLA-GOLD` aktif.
2. Alokasikan unit cadangan setara dari stok **Buffer Warehouse Regional** maksimal dalam waktu 4 jam kerja sejak tiket diverifikasi.
3. Cetak Berita Acara Peminjaman Perangkat (`BAST-LOANER`) dan jadwalkan kurir ekspres atau teknisi lapangan untuk menukar unit rusak dengan unit pinjaman di lokasi klien.
4. Pantau masa peminjaman unit cadangan di dashboard SIDIG hingga unit utama selesai diperbaiki dan dikembalikan.

### Batas Eskalasi Teknis SLA-GOLD
Apabila stok unit cadangan di gudang regional kosong, segera aktifkan protokol eskalasi logistik antar-cabang dengan persetujuan Koordinator Operasional SIDIG.

## 2. Eskalasi Penggantian Mainboard yang Melewati Batas SLA 14 Hari Kerja (RMA-19)
### Gejala & Indikasi Teknis RMA-19
Tiket perbaikan perangkat di pusat servis purnajual SIDIG memunculkan peringatan keterlambatan `RMA-19` (Mainboard Replacement SLA Breach > 14 Working Days). Penggantian papan induk (mainboard) tertunda akibat kelangkaan komponen chipset dari pemasok sehingga melewati batas komitmen SLA 14 hari kerja.

### Langkah Penanganan & Kompensasi RMA-19
1. Periksa status antrean suku cadang mainboard pada sistem inventaris **SIDIG Sparepart Supply Chain** menggunakan nomor tiket RMA dan nomor seri `SN-xxxxx`.
2. Apabila estimasi kedatangan suku cadang mainboard melebihi 3 hari kerja ke depan, ajukan opsi **Unit Swap / Penggantian Unit Rekondisi Bersertifikat (Certified Refurbished Replacement)** kepada pelanggan.
3. Berikan kompensasi perpanjangan masa garansi resmi selama **+3 bulan tambahan** secara otomatis pada nomor seri perangkat pelanggan di portal SIDIG.
4. Kirimkan pemberitahuan resmi permohonan maaf dan jadwal pengiriman baru yang telah disetujui supervisor purnajual.

### Batas Eskalasi Teknis RMA-19
Seluruh tiket dengan status `RMA-19` yang belum terselesaikan pada hari kerja ke-16 wajib dieskalasikan langsung ke Kepala Divisi After-Sales & Quality Assurance.
