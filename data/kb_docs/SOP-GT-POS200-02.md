# SOP-GT-POS200-02: Prosedur Rekonsiliasi Settlement Harian dan Pemulihan Database SQLite Lokal GT-POS200
Product Model: GT-POS200
Category: Billing_Payment

## 1. Rekonsiliasi Transaksi Settlement Harian yang Menggantung (E-501)
### Gejala & Indikasi Teknis E-501
Pada saat penutupan kasir akhir hari (End-of-Day Closing), terminal GT-POS200 gagal menyelesaikan proses pengiriman batch transaksi dan menampilkan kode kesalahan `E-501` (Daily Settlement Batch Stuck / Host Reconciliation Mismatch). Status transaksi di layar kasir tertulis `PENDING_SETTLEMENT` sehingga laporan omzet harian toko tidak seimbang dengan mutasi bank.

### Langkah Perbaikan Standar E-501
1. Masuk ke menu **Manajemen Kasir > Rekonsiliasi & Settlement > Audit Batch Menggantung** menggunakan PIN Supervisor Toko.
2. Pilih tombol **Sinkronisasi Ulang Jurnal Transaksi (Force Batch Inquiry)** untuk mencocokkan ID referensi lokal (`STAN`) dengan server payment gateway pusat.
3. Periksa daftar transaksi yang berstatus `UNCONFIRMED`; apabila dana telah terpotong di sisi pelanggan, tekan **Tandai Terverifikasi & Cetak Bukti Rekonsiliasi**.
4. Jalankan ulang prosedur **Cut-Off Settlement Harian** dan simpan berkas log `settlement_audit_E501.json` sebagai lampiran laporan keuangan.

### Batas Eskalasi Teknis E-501
Jika selisih nominal settlement melebihi Rp 500.000 atau status batch tidak berubah setelah 30 menit, tahan penutupan buku kasir dan eskalasikan ke divisi Billing & Payment Gateway Operations.

## 2. Pemulihan Basis Data SQLite Lokal Akibat Mati Listrik Mendadak (E-509)
### Gejala & Indikasi Teknis E-509
Setelah insiden listrik padam mendadak atau baterai cadangan habis total, aplikasi kasir GT-POS200 gagal masuk ke halaman penjualan utama dan menampilkan kode `E-509` (Local SQLite Database Corrupt / WAL Journal Lock). Pesan detail menunjukkan `database disk image is malformed` pada berkas `pos_transactions.db`.

### Langkah Perbaikan Standar E-509
1. Masuk ke mode pemulihan sistem (Recovery Shell) dengan menahan tombol **F2 + Power** saat terminal GT-POS200 dinyalakan.
2. Pilih utilitas **Perbaikan Basis Data Lokal (SQLite Integrity & WAL Checkpoint Recovery)**.
3. Sistem secara otomatis mencadangkan berkas rusak ke `pos_transactions.db.bak`, mengeksekusi perintah `PRAGMA wal_checkpoint(TRUNCATE);`, dan membangun ulang indeks tabel transaksi dari jurnal cadangan (`shadow_journal.log`).
4. Setelah status integritas menunjukkan `OK`, mulai ulang aplikasi kasir dan verifikasi 5 transaksi terakhir sebelum listrik padam.

### Batas Eskalasi Teknis E-509
Apabila utilitas otomatis gagal memulihkan tabel utama karena kerusakan sektor penyimpanan flash, lakukan sinkronisasi tarik data dari Cloud Backup GT-POS dan segera jadwalkan pemeriksaan media penyimpanan.
