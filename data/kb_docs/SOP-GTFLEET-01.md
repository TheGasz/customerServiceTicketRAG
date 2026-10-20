# SOP-GTFLEET-01: Diagnosis Kehilangan Sinyal Antena GNSS/GPS (E-901) dan Kalibrasi Sensor Bahan Bakar & Kartu Tap Armada (E-904)
Product Model: GTFLEET-TRACKER
Category: Hardware_POS

## 1. Diagnosis Kehilangan Sinyal Antena GNSS/GPS pada Unit Telemetri Armada Bus (E-901)
### Gejala & Indikasi Teknis E-901
Unit telemetri kendaraan **GTFLEET-TRACKER** yang terpasang pada armada bus atau truk logistik berhenti mengirimkan pembaruan titik koordinat lokasi di peta pusat kendali dan memunculkan kode `E-901` (GNSS/GPS Antenna Open Circuit or Satellite Fix Lost). Indikator status satelit pada dashboard tertulis `NO_FIX (0 Satellites)` meskipun armada sedang beroperasi di jalan terbuka.

### Langkah Perbaikan Standar E-901
1. Buka **GTFLEET Command Center > Diagnostik Perangkat Armada**, masukkan nomor seri perangkat `SN-xxxxx` atau nomor lambung bus untuk memeriksa telemetri perangkat keras jarak jauh.
2. Periksa parameter `gnss_antenna_impedance`; jika bernilai `OPEN_CIRCUIT` atau tegangan arus antena 0 mA, berarti konektor SMA antena GPS eksternal di dashboard kendaraan kendor atau kabel terjepit.
3. Kirimkan perintah jarak jauh **Remote Cold-Start GNSS Module & Ephemeris Assistance Update (A-GPS)** melalui konsol telemetri.
4. Jika sinyal tetap tidak terkunci setelah 10 menit, instruksikan teknisi depo armada untuk mengencangkan konektor baut SMA kuningan pada port `ANT-GPS` dan memindahkan modul penerima ke area kaca depan yang tidak terhalang kaca film logam.

### Batas Eskalasi Teknis E-901
Apabila lebih dari 5 unit armada bus kota kehilangan sinyal `E-901` secara bersamaan, periksa gangguan layanan A-GPS server pusat atau jadwalkan penggantian antena aktif di depo utama.

## 2. Kalibrasi Sensor Bahan Bakar Ultrasonik dan Pembaca Kartu Tap Penumpang (E-904)
### Gejala & Indikasi Teknis E-904
Sistem telemetri **GTFLEET-TRACKER** menampilkan kode peringatan `E-904` (Fuel Level Sensor Drift / Passenger Tap Reader RS-485 Checksum Error). Gejala yang terlihat adalah grafik volume solar di dashboard melonjak turun-naik secara tidak wajar saat bus melewati jalan bergelombang, atau terminal tap kartu elektronik penumpang di pintu bus lambat merespons.

### Langkah Perbaikan Standar E-904
1. Masuk ke menu **Konfigurasi Periferal Armada > Bus RS-485 & Kalibrasi Sensor** pada portal GTFLEET.
2. Aktifkan filter peredam guncangan **Kalman Smoothing Filter (Window = 60 detik)** pada kanal pembacaan sensor bahan bakar ultrasonik untuk meredam fluktuasi permukaan tangki saat armada bergerak.
3. Untuk gangguan pembaca kartu tap penumpang, periksa kecepatan komunikasi serial `baud_rate = 115200` pada port RS-485 dan tekan tombol **Reset RS-485 Transceiver & Sync Tariff Table**.
4. Lakukan uji tap menggunakan kartu uji koridor bus dan pastikan waktu respons validasi di bawah 300 milidetik.

### Batas Eskalasi Teknis E-904
Jika ditemukan indikasi pencurian bahan bakar nyata atau kerusakan fisik kabel sensor di bawah sasis bus, kirimkan tiket pemeriksaan fisik ke tim Mekanik & Teknisi Depo Armada.
