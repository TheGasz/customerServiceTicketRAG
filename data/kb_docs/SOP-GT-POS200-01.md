# SOP-GT-POS200-01: Panduan Perbaikan Modul Pembaca QRIS/NFC dan Printer Termal Terminal Kasir GT-POS200
Product Model: GT-POS200
Category: Hardware_POS

## 1. Kalibrasi Ulang Modul Pembaca QRIS dan NFC Gagal Inisialisasi (E-402)
### Gejala & Indikasi Teknis E-402
Terminal kasir elektronik GT-POS200 menampilkan pesan kesalahan `E-402` (QRIS/NFC Reader Initialization Failure) saat kasir memindai kode pembayaran QRIS dinamis atau kartu pembayaran nirsentuh (NFC contactless). Lampu indikator LED pada pemindai optik berkedip merah tiga kali dan layar menampilkan peringatan modul optik tidak merespons pada bus komunikasi I2C.

### Langkah Perbaikan Standar E-402
1. Buka menu **Pengaturan Perangkat > Diagnostik Periferal > Modul Optik & NFC** pada layar GT-POS200.
2. Tekan tombol **Reset Bus I2C & Kalibrasi Sensor QRIS** lalu tunggu proses pengosongan kapasitor selama 15 detik.
3. Bersihkan jendela kaca pemindai QRIS menggunakan kain mikrofiber kering bebas serat untuk menghilangkan noda minyak atau debu kasir.
4. Lakukan pemindaian kartu uji kalibrasi (Test Card GT-CAL-01); pastikan tegangan pembacaan berada pada rentang 3.25V - 3.35V.
5. Jika kode `E-402` tetap muncul setelah dua kali kalibrasi ulang, cabut konektor pita fleksibel modul NFC di bagian bawah casing dan pasang kembali dengan pengunci rapat.

### Batas Eskalasi Teknis E-402
Apabila seluruh gerai kasir tidak dapat menerima pembayaran non-tunai pada jam operasional puncak, eskalasikan tiket ke tim teknisi lapangan (On-Site Hardware Support) atau ajukan klaim RMA penggantian modul optik.

## 2. Pembersihan Sensor Kepala Cetak Termal dan Penanganan Kertas Macet (E-405)
### Gejala & Indikasi Teknis E-405
Layar terminal GT-POS200 memunculkan kode `E-405` (Thermal Printhead Overheat / Paper Jam Sensor Fault) ketika mencetak struk transaksi belanja pelanggan. Hasil cetakan struk tampak bergaris putih putus-putus, pudar sebelah, atau motor penggulung kertas berhenti mendadak disertai bunyi bip panjang.

### Langkah Perbaikan Standar E-405
1. Matikan daya unit GT-POS200 melalui sakelar utama dan biarkan kepala cetak (thermal printhead) mendingin selama minimal 3 menit.
2. Buka penutup kompartemen gulungan kertas termal 58mm, keluarkan sisa potongan kertas struk yang menyangkut pada poros roller karet (platen roller).
3. Gunakan kapas bertangkai yang dibasahi alkohol isopropil (IPA 90%) untuk mengusap permukaan garis pemanas kepala cetak termal dan sensor optik pendeteksi kertas habis.
4. Pastikan gulungan kertas termal dipasang dengan posisi sisi peka panas menghadap ke kepala cetak, lalu jalankan perintah **Self-Test Print** dari menu pemeliharaan.

### Batas Eskalasi Teknis E-405
Jangan pernah menggunakan benda tajam atau obeng logam untuk mencongkel kerak karbon pada kepala cetak termal karena akan merusak lapisan pelindung elemen pemanas dan membatalkan garansi perangkat.
