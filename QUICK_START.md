# ⚡ Quick Start - Civitas UB Detection

## 🚀 Langkah Super Cepat (2 Menit)

### **Opsi 1: One-Click Setup (Termudah)**
```bash
# Windows: Double-click file ini
run-civitas.bat

# Linux/Mac: Jalankan script ini
./run-civitas.sh
```

### **Opsi 2: Manual Setup**
```bash
# 1. Install dependencies
pip install -r requirements-civitas.txt

# 2. Jalankan program
python main-jetson-civitas.py
```

## 📋 Yang Dibutuhkan

- [ ] Python 3.7+ terinstall
- [ ] Webcam/kamera tersedia
- [ ] File project lengkap

## 🎯 Hasil yang Diharapkan

### ✅ **Civitas UB Terdeteksi**
- Kotak wajah berwarna **GOLD**
- Label: **"Person | UB"**
- Dashboard: **"Civitas UB"** dengan confidence tinggi

### ❓ **Status Tidak Pasti**
- Kotak wajah berwarna **ORANGE**
- Label: **"Person | ?"**
- Dashboard: **"Uncertain"**

### ❌ **Non-Civitas UB**
- Kotak wajah berwarna **GRAY**
- Label: **"Person"**
- Dashboard: **"Non-Civitas UB"**

## 🔧 Jika Ada Masalah

### **Kamera tidak buka**
```bash
# Coba ganti index kamera
# Edit line: cv2.VideoCapture(0) menjadi cv2.VideoCapture(1)
```

### **Error import**
```bash
pip install -r requirements-civitas.txt
```

### **Template tidak ditemukan**
- Pastikan folder `templates/` ada
- Download logo UB (colored & bw)

## 📞 Bantuan Cepat

**Program berjalan tapi tidak akurat?**
- Perbaiki pencahayaan
- Pastikan logo UB terlihat
- Tunggu 3-5 detik untuk stabilisasi

**Keluar dari program:**
- Tekan tombol **'q'**

---
**Total waktu setup: ~2 menit | One-click setup tersedia!**