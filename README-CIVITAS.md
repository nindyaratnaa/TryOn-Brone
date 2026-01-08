# 🎯 Civitas UB Detection System

Sistem deteksi otomatis untuk mengidentifikasi **Civitas Universitas Brawijaya** menggunakan computer vision.

## ⚡ Quick Start (2 Menit)

### **Windows Users**
```bash
# Double-click file ini
run-civitas.bat
```

### **Linux/Mac Users**
```bash
# Jalankan script ini
chmod +x run-civitas.sh
./run-civitas.sh
```

### **Manual Setup**
```bash
pip install -r requirements-civitas.txt
python main-jetson-civitas.py
```

## 📁 Struktur Project

```
TryOn-Brone/
├── main-jetson-civitas.py      # Program utama
├── requirements-civitas.txt    # Dependencies
├── run-civitas.bat            # One-click setup Windows
├── run-civitas.sh             # One-click setup Linux/Mac
├── QUICK_START.md             # Panduan cepat
├── USER_GUIDE_CIVITAS.md      # Panduan lengkap
├── haarcascades/              # Face detection model
└── templates/                 # Logo UB templates
```

## 🎯 Fitur

- ✅ **Real-time Detection**: Deteksi civitas secara langsung
- ✅ **Logo Recognition**: Mengenali logo UB pada pakaian
- ✅ **Color Detection**: Deteksi warna almamater navy
- ✅ **Temporal Smoothing**: Hasil stabil dengan averaging
- ✅ **Visual Dashboard**: Interface informatif
- ✅ **High Performance**: 25-35 FPS

## 🛠️ Requirements

- Python 3.7+
- Webcam/Camera
- OpenCV 4.5+
- NumPy 1.19+

## 📖 Dokumentasi

- **`QUICK_START.md`** - Setup dalam 2 menit
- **`USER_GUIDE_CIVITAS.md`** - Panduan lengkap & troubleshooting

## 🎬 Cara Menggunakan

1. **Jalankan program** (gunakan script one-click)
2. **Posisi di depan kamera** (jarak 1-2 meter)
3. **Tunggu deteksi wajah** (kotak akan muncul)
4. **Lihat hasil** di dashboard kiri atas
5. **Tekan 'q'** untuk keluar

## 🎨 Status Detection

- 🟡 **Gold Box**: Civitas UB terdeteksi
- 🟠 **Orange Box**: Status tidak pasti  
- ⚫ **Gray Box**: Non-Civitas UB

## 🔧 Troubleshooting

**Kamera tidak buka?**
- Pastikan kamera terhubung
- Tutup aplikasi lain yang pakai kamera

**Deteksi tidak akurat?**
- Perbaiki pencahayaan
- Pastikan logo UB terlihat jelas
- Tunggu 3-5 detik untuk stabilisasi

## 📞 Support

Lihat file `USER_GUIDE_CIVITAS.md` untuk panduan lengkap dan troubleshooting detail.

---
**Developed for Universitas Brawijaya | Ready to use in 2 minutes!**