# SOP-GTCAMPUS-01: Pembukaan Kunci Pengisian KRS Mahasiswa (ERR_KRS_LOCKED) dan Sinkronisasi Token SSO Kampus (SSO-401)
Product Model: GTCAMPUS-SIAKAD
Category: Academic_Portal

## 1. Pembukaan Kunci Pengisian KRS Mahasiswa Pasca-Verifikasi Dosen Wali (ERR_KRS_LOCKED)
### Gejala & Indikasi Teknis ERR_KRS_LOCKED
Mahasiswa atau operator akademik melaporkan bahwa halaman Kartu Rencana Studi (KRS) pada portal **GTCAMPUS-SIAKAD** tidak dapat diubah atau disimpan dengan pesan kesalahan `ERR_KRS_LOCKED` (Study Plan Modification Locked After Advisor Approval Window). Hal ini terjadi ketika dosen pembimbing akademik telah menekan tombol persetujuan atau terjadi kegagalan pembaruan flag status masa revisi KRS.

### Langkah Perbaikan Standar ERR_KRS_LOCKED
1. Masuk ke panel **Admin Akademik GTCAMPUS-SIAKAD > Manajemen KRS > Status Kunci Mahasiswa** dan cari data mahasiswa menggunakan NIM atau ID pengguna.
2. Periksa atribut `krs_approval_status` dan `revision_window_active` pada semester berjalan.
3. Apabila masa batal-tambah (revisi KRS) resmi kampus masih berlangsung dan dosen wali telah memberikan izin revisi, klik tombol **Reset Flag Kunci KRS (Unlock for Revision)** untuk mengubah status dari `FINAL_LOCKED` menjadi `DRAFT_REVISION`.
4. Bersihkan cache sesi akademik mahasiswa pada Redis Cache Server melalui tombol **Purge Student Session Cache** agar perubahan langsung terlihat di portal mahasiswa.

### Batas Eskalasi Teknis ERR_KRS_LOCKED
Apabila kode `ERR_KRS_LOCKED` dialami secara massal oleh seluruh mahasiswa satu fakultas saat jam sibuk pengisian KRS, segera eskalasikan ke tim Database Administrator (DBA) karena indikasi deadlock pada tabel `akad_krs_mahasiswa`.

## 2. Sinkronisasi Token Otentikasi Single Sign-On Kampus (SSO-401)
### Gejala & Indikasi Teknis SSO-401
Pengguna (mahasiswa, dosen, atau staf akademik) gagal masuk ke portal **GTCAMPUS-SIAKAD** dan terus diarahkan berulang kali (redirect loop) ke halaman login dengan kode kesalahan `SSO-401` (CAS/OAuth2 Token Signature Expired or Clock Skew Mismatch).

### Langkah Perbaikan Standar SSO-401
1. Buka konsol **GTCAMPUS Identity & SSO Gateway > Konfigurasi OAuth2 / OIDC Provider**.
2. Periksa sinkronisasi waktu protokol NTP (Network Time Protocol) antara server aplikasi SIAKAD dan server pusat Identity Provider (IdP); pastikan selisih waktu (clock skew) kurang dari 5 detik.
3. Jika sertifikat penandatanganan JWT baru saja diperbarui oleh tim TI kampus, tekan tombol **Reload JWKS Public Key & Clear Token Cache**.
4. Instruksikan pengguna untuk menghapus cookie domain `.kampus.ac.id` pada peramban atau mencoba login ulang melalui jendela penyamaran (Incognito Mode).

### Batas Eskalasi Teknis SSO-401
Jika kegagalan otentikasi `SSO-401` berdampak pada portal ujian online atau penutupan nilai akhir semester, prioritaskan sebagai insiden tinggi dan hubungkan langsung dengan tim Infrastruktur Keamanan Jaringan.
