# SOP-GTCAMPUS-02: Penanganan Callback Webhook Pembayaran UKT/SPP Virtual Account (ERR_VA_TIMEOUT) dan Rekonsiliasi Buku Besar (VA-808)
Product Model: GTCAMPUS-SIAKAD
Category: Billing_Payment

## 1. Penanganan Callback Webhook Pembayaran UKT/SPP Virtual Account Terlambat Masuk (ERR_VA_TIMEOUT)
### Gejala & Indikasi Teknis ERR_VA_TIMEOUT
Mahasiswa telah melunasi tagihan Uang Kuliah Tunggal (UKT) atau SPP melalui Virtual Account bank mitra dan saldo rekening telah terdebet, namun status tagihan di portal **GTCAMPUS-SIAKAD** masih tertulis `BELUM_LUNAS` dengan log sistem mencatat `ERR_VA_TIMEOUT` (Bank Webhook Callback Timeout / Unacknowledged Payment Notification). Akibatnya, mahasiswa belum dapat mengakses pengisian KRS.

### Langkah Perbaikan Standar ERR_VA_TIMEOUT
1. Buka modul **GTCAMPUS Finance & Billing > Monitoring Virtual Account > Log Callback Bank Mitra**.
2. Masukkan Nomor Virtual Account atau Nomor Induk Mahasiswa (NIM) untuk memeriksa antrean pesan webhook yang berstatus `TIMEOUT_RETRY`.
3. Tekan tombol **Check Payment Status via Bank Inquiry API (Pull Reconciliation)** untuk menarik status pelunasan langsung dari server bank mitra secara real-time.
4. Begitu respons bank menunjukkan kode `PAID (00)`, sistem otomatis memperbarui status tagihan UKT menjadi `LUNAS`, menerbitkan kuitansi elektronik, dan membuka blokir hak akses KRS mahasiswa.

### Batas Eskalasi Teknis ERR_VA_TIMEOUT
Apabila jalur koneksi API Inquiry ke bank mitra mengalami gangguan nasional atau sertifikat mTLS kedaluwarsa pada hari terakhir pembayaran UKT, segera eskalasikan ke tim Integrasi Perbankan & Bagian Keuangan Universitas.

## 2. Prosedur Rekonsiliasi Selisih Buku Besar Tagihan Akademik (VA-808)
### Gejala & Indikasi Teknis VA-808
Operator keuangan universitas menemukan ketidakcocokan jumlah pembayaran pada laporan harian dengan kode peringatan `VA-808` (Academic Ledger Double-Posting or Partial Tariff Mismatch). Hal ini terjadi ketika perubahan skema keringanan/cicilan UKT diterapkan setelah nomor Virtual Account diterbitkan ke bank.

### Langkah Perbaikan Standar VA-808
1. Masuk ke menu **Keuangan Akademik > Buku Besar & Rekonsiliasi > Audit Anomali VA-808**.
2. Bandingkan nilai `nominal_tagihan_siakad` dengan `nominal_terbayar_bank` pada rincian invoice mahasiswa.
3. Jika terjadi kelebihan atau kekurangan pencatatan akibat pembaruan skema cicilan UKT, jalankan fungsi **Sinkronisasi Ulang Tarif & Jurnal Penyesuaian Otomatis (Ledger Re-Sync)**.
4. Batalkan nomor Virtual Account lama yang tidak valid dan terbitkan nomor Virtual Account baru sesuai nominal tagihan revisi yang telah disahkan Biro Keuangan.

### Batas Eskalasi Teknis VA-808
Perubahan nominal tagihan masa lalu yang telah masuk ke laporan audit tahunan memerlukan persetujuan otorisasi berlapis dari Kepala Biro Keuangan Kampus.
