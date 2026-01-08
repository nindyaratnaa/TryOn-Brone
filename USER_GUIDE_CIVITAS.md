# 📖 Panduan Penggunaan Sistem Deteksi Civitas UB

## 🎯 Apa itu Sistem Ini?
Sistem deteksi otomatis untuk mengidentifikasi **Civitas Universitas Brawijaya** berdasarkan:
- Logo UB pada pakaian
- Warna almamater navy (biru dongker)
- Kombinasi keduanya

## 🛠️ Persiapan Sebelum Menggunakan

### 1. **Persyaratan Hardware**
- Komputer/Laptop dengan webcam ATAU
- Jetson Nano/Xavier dengan CSI camera/USB webcam
- RAM minimal 4GB
- Python 3.7+

### 2. **Install Dependencies**
```bash
pip install opencv-python numpy
```

### 3. **Download File yang Diperlukan**
Pastikan folder project memiliki struktur:
```
TryOn-Brone/
├── main-jetson-civitas.py          # File utama
├── haarcascades/
│   └── haarcascade_frontalface_default.xml
└── templates/
    ├── ub_logo_colored.png         # Logo UB berwarna
    └── ub_logo_bw.png             # Logo UB hitam putih
```

## 🚀 Cara Menjalankan

### **Langkah 1: Buka Terminal/Command Prompt**
```bash
cd /path/to/TryOn-Brone
```

### **Langkah 2: Jalankan Program**
```bash
python main-jetson-civitas.py
```

### **Langkah 3: Sistem Siap Digunakan**
- Jendela kamera akan terbuka
- Posisikan diri di depan kamera
- Sistem akan otomatis mendeteksi wajah dan status civitas

## 📺 Cara Menggunakan Interface

### **Dashboard Informasi (Kiri Atas)**
- **Instant Civitas**: Hasil deteksi real-time
- **Smoothed Civitas**: Hasil yang sudah di-rata-rata (lebih stabil)
- **Civitas Buffer**: Jumlah data yang tersimpan untuk averaging
- **FPS**: Frame per second (kecepatan pemrosesan)

### **Indikator Visual**
- **Kotak Wajah**:
  - 🟡 **Gold**: Civitas UB terdeteksi
  - 🟠 **Orange**: Status tidak pasti
  - ⚫ **Gray**: Non-Civitas UB

- **Kotak Chest Area**: Area deteksi logo/almamater
- **Kotak Logo UB**: Lokasi logo yang terdeteksi

### **Label Status**
- **"Person | UB"**: Civitas UB
- **"Person | ?"**: Status tidak pasti
- **"Person"**: Non-Civitas UB

## ⚙️ Tips Penggunaan Optimal

### **Posisi Terbaik**
1. **Jarak**: 1-2 meter dari kamera
2. **Pencahayaan**: Cukup terang, hindari backlight
3. **Posisi**: Menghadap kamera, badan tegak
4. **Pakaian**: Logo UB terlihat jelas di dada

### **Untuk Hasil Akurat**
- Tunggu 3-5 detik untuk hasil yang stabil
- Pastikan logo UB tidak tertutup
- Hindari gerakan terlalu cepat
- Gunakan pencahayaan yang merata

## 🎨 Memahami Hasil Deteksi

### **Status Civitas UB** (Confidence > 0.6)
- Logo UB terdeteksi jelas
- Warna navy almamater teridentifikasi
- Kombinasi keduanya memenuhi threshold

### **Status Uncertain** (Confidence 0.4-0.6)
- Logo UB samar atau parsial
- Warna navy terdeteksi tapi tidak optimal
- Butuh posisi/pencahayaan lebih baik

### **Status Non-Civitas UB** (Confidence < 0.4)
- Tidak ada logo UB yang terdeteksi
- Warna pakaian bukan navy UB
- Bukan civitas Universitas Brawijaya

## 🔧 Troubleshooting

### **Problem: Kamera tidak terbuka**
```
❌ Gagal membuka webcam.
```
**Solusi**:
- Pastikan kamera terhubung
- Tutup aplikasi lain yang menggunakan kamera
- Coba ganti index kamera: `cv2.VideoCapture(1)` atau `cv2.VideoCapture(2)`

### **Problem: Template logo tidak ditemukan**
```
⚠ Template not found: templates/ub_logo_colored.png
```
**Solusi**:
- Pastikan folder `templates/` ada
- Download file logo UB yang diperlukan
- Periksa nama file sesuai dengan konfigurasi

### **Problem: Deteksi tidak akurat**
**Solusi**:
- Perbaiki pencahayaan
- Pastikan logo UB terlihat jelas
- Tunggu beberapa detik untuk temporal averaging
- Sesuaikan jarak dari kamera

### **Problem: FPS rendah**
**Solusi**:
- Tutup aplikasi lain yang berat
- Kurangi resolusi kamera jika perlu
- Pastikan hardware memadai

## ⌨️ Kontrol Program

- **'q'**: Keluar dari program
- **ESC**: Alternatif untuk keluar
- **Spacebar**: (Opsional) Screenshot hasil

## 📊 Spesifikasi Teknis

### **Akurasi Deteksi**
- Logo UB: Template matching dengan threshold 0.6
- Warna Navy: HSV color detection
- Kombinasi: Rule-based classification

### **Performance**
- FPS: 25-35 (tergantung hardware)
- Latency: < 100ms per frame
- Memory: ~200MB RAM

### **Supported Formats**
- Input: BGR video stream
- Resolution: 1280x720 (dapat disesuaikan)
- Color space: BGR, HSV, Grayscale

## 📞 Support & Bantuan

Jika mengalami masalah:
1. Periksa troubleshooting di atas
2. Pastikan semua file diperlukan ada
3. Cek versi Python dan OpenCV
4. Restart program jika perlu

---

**Sistem ini dikembangkan untuk membantu identifikasi civitas Universitas Brawijaya secara otomatis dan real-time.**