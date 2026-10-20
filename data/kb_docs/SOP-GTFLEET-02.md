# SOP-GTFLEET-02: Pemulihan Koneksi Broker Telemetri MQTT (ERR_MQTT_DROP) dan Rollback Firmware Over-The-Air Gagal (OTA-702)
Product Model: GTFLEET-TRACKER
Category: Hardware_POS

## 1. Pemulihan Koneksi Broker Telemetri MQTT yang Terputus Berulang (ERR_MQTT_DROP)
### Gejala & Indikasi Teknis ERR_MQTT_DROP
Perangkat IoT **GTFLEET-TRACKER** mengalami putus-nyambung (flapping connection) ke server pusat dan mencatat kode kesalahan `ERR_MQTT_DROP` (MQTT Broker Keep-Alive Timeout / TLS Handshake Rejected). Data telemetri kecepatan dan posisi kendaraan tertahan di memori buffer lokal perangkat dan baru terkirim secara menumpuk beberapa jam kemudian.

### Langkah Perbaikan Standar ERR_MQTT_DROP
1. Buka dashboard **GTFLEET IoT Core > Broker MQTT & Manajemen Koneksi Seluler** dan lacak nomor seri unit `SN-xxxxx`.
2. Periksa kualitas sinyal kartu SIM M2M (`RSSI` dan `Packet Loss`); jika unit berpindah antar-menara BTS di jalur luar kota, ubah parameter `mqtt_keepalive_interval` dari 15 detik menjadi **60 detik** dan aktifkan mode `QoS = 1 (At Least Once)`.
3. Apabila log menunjukkan kegagalan `TLS Client Certificate Expired`, jalankan perintah **Push Secondary TLS Credential via SMS Command Fallback** ke nomor kartu M2M perangkat.
4. Setelah koneksi MQTT stabil kembali, verifikasi bahwa antrean data tertunda pada buffer microSD lokal (`offline_telemetry_spool.bin`) telah terkirim seluruhnya ke server.

### Batas Eskalasi Teknis ERR_MQTT_DROP
Jika pemutusan koneksi `ERR_MQTT_DROP` terjadi serentak pada ratusan unit telemetri di seluruh wilayah, segera eskalasikan ke tim Cloud DevOps untuk memeriksa beban klaster EMQX/Mosquitto Broker.

## 2. Prosedur Rollback Firmware Over-The-Air yang Gagal atau Bootloop (OTA-702)
### Gejala & Indikasi Teknis OTA-702
Setelah proses pembaruan perangkat lunak jarak jauh (Over-The-Air Update), unit **GTFLEET-TRACKER** gagal memuat sistem operasi utama dan melaporkan kode kesalahan `OTA-702` (OTA Firmware Checksum Mismatch / Dual-Bank Bootloader Fallback Triggered). Unit terjebak dalam mode darurat (Safe Mode) sehingga fitur pelacakan lanjutan tidak aktif.

### Langkah Perbaikan Standar OTA-702
1. Masuk ke konsol **GTFLEET Firmware Management > Dual-Bank Bootloader Control** dan pilih perangkat yang berstatus `OTA-702_SAFE_MODE`.
2. Eksekusi perintah **Trigger Partition B Rollback (Restore Previous Stable Firmware v3.4.2)** untuk mengembalikan perangkat ke partisi firmware stabil sebelumnya secara instan.
3. Bersihkan direktori unduhan sementara `/ota_staging/` pada memori flash perangkat agar paket firmware yang korup akibat sinyal seluler terputus dihapus sepenuhnya.
4. Setelah unit kembali online dengan versi stabil, jadwalkan ulang pembaruan OTA hanya ketika kendaraan sedang parkir di depo dengan koneksi Wi-Fi depo yang stabil.

### Batas Eskalasi Teknis OTA-702
Jangan pernah memaksakan perintah flash ulang melalui jaringan seluler 2G/EDGE yang lemah karena berisiko menghabiskan kuota M2M dan membuat watchdog timer merestart perangkat terus-menerus.
