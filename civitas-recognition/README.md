# Civitas Recognition Module

Modul deteksi civitas Universitas Brawijaya yang terpisah dari sistem emotion recognition utama.

## Struktur Modul

```
civitas-recognition/
├── __init__.py          # Main module exports
├── config.py            # Konfigurasi civitas detection
├── detector.py          # CivitasDetector class
├── temporal_averaging.py # CivitasTemporalAveraging class
├── ui_utils.py          # CivitasUI utilities
└── README.md           # Dokumentasi ini
```

## Komponen Utama

### 1. CivitasConfig
Konfigurasi untuk sistem deteksi civitas:
- Template logo UB
- Definisi warna HSV untuk navy dan gold UB
- Threshold deteksi
- Parameter temporal averaging

### 2. CivitasDetector
Class utama untuk deteksi civitas UB:
- `detect_ub_logo()`: Deteksi logo UB menggunakan template matching dan color detection
- `detect_civitas_status()`: Klasifikasi civitas berdasarkan almamater dan logo

### 3. CivitasTemporalAveraging
Smoothing hasil deteksi civitas:
- Buffer untuk menyimpan prediksi
- Averaging untuk hasil yang stabil
- Status: "Civitas UB", "Non-Civitas UB", "Uncertain"

### 4. CivitasUI
Utilities untuk menampilkan hasil deteksi:
- `draw_civitas_info()`: Menampilkan info civitas di dashboard
- `draw_civitas_detection_boxes()`: Menggambar kotak deteksi
- `add_civitas_label_to_face()`: Menambah label civitas ke wajah

## Cara Penggunaan

```python
from civitas_recognition import CivitasDetector, CivitasTemporalAveraging, CivitasUI

# Inisialisasi
detector = CivitasDetector()
temporal_avg = CivitasTemporalAveraging()

# Deteksi civitas
status, confidence, chest_box, logo_box = detector.detect_civitas_status(frame, x, y, w, h)

# Update temporal averaging
is_civitas = status == "Civitas UB"
temporal_avg.add_prediction(confidence, is_civitas)
smooth_status, smooth_conf = temporal_avg.get_averaged_civitas()

# Tampilkan hasil
CivitasUI.draw_civitas_info(frame, x, y, spacing, 
                           (status, confidence), 
                           (smooth_status, smooth_conf))
```

## Fitur Deteksi

1. **Template Matching**: Mencocokkan logo UB dengan template
2. **Color Detection**: Deteksi warna navy dan gold khas UB
3. **Geometric Pattern**: Deteksi pola lingkaran logo UB
4. **Almamater Detection**: Deteksi warna navy almamater UB
5. **Temporal Smoothing**: Hasil yang stabil dengan averaging

## Dependencies

- OpenCV (cv2)
- NumPy
- Collections (deque)

## Integrasi

Modul ini terintegrasi dengan:
- `main.py` - Sistem FER utama dengan PyTorch
- `main-jetson.py` - Sistem FER untuk Jetson dengan ONNX