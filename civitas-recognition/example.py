"""
Contoh penggunaan Civitas Recognition Module
"""
import cv2
import numpy as np
from civitas_recognition import CivitasDetector, CivitasTemporalAveraging, CivitasUI

def example_civitas_detection():
    """Contoh sederhana penggunaan civitas detection"""
    
    # 1. Inisialisasi komponen
    detector = CivitasDetector()
    temporal_avg = CivitasTemporalAveraging()
    
    # 2. Setup camera (contoh)
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("Civitas Detection Demo - Tekan 'q' untuk keluar")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        
        if len(faces) > 0:
            # Ambil wajah terbesar
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            
            # 3. Deteksi civitas status
            instant_status, instant_conf, chest_box, logo_box = detector.detect_civitas_status(frame, x, y, w, h)
            
            # 4. Update temporal averaging
            is_civitas = instant_status == "Civitas UB"
            temporal_avg.add_prediction(instant_conf, is_civitas)
            smooth_status, smooth_conf = temporal_avg.get_averaged_civitas()
            
            # 5. Gambar hasil deteksi
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Gambar civitas detection boxes
            CivitasUI.draw_civitas_detection_boxes(frame, chest_box, logo_box, 
                                                  instant_conf, smooth_status)
            
            # Gambar info civitas
            CivitasUI.draw_civitas_info(frame, 20, 50, 30,
                                       (instant_status, instant_conf),
                                       (smooth_status, smooth_conf))
            
            # Label pada wajah
            label = CivitasUI.add_civitas_label_to_face("Person", (smooth_status, smooth_conf))
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Buffer info
            buffer_info = temporal_avg.get_buffer_info()
            cv2.putText(frame, f"Buffer: {buffer_info['length']}/{buffer_info['capacity']} ({buffer_info['percentage']}%)", 
                       (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        else:
            cv2.putText(frame, "Mencari wajah...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            temporal_avg.reset()
        
        cv2.imshow('Civitas Detection Demo', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def test_civitas_config():
    """Test konfigurasi civitas"""
    from civitas_recognition.config import CivitasConfig
    
    print("=== Civitas Configuration ===")
    print(f"Labels: {CivitasConfig.CIVITAS_LABELS}")
    print(f"Logo Templates: {CivitasConfig.UB_LOGO_TEMPLATES}")
    print(f"UB Colors: {CivitasConfig.UB_COLORS}")
    print(f"Window Size: {CivitasConfig.CIVITAS_WINDOW_SIZE}")
    print(f"Confidence Threshold: {CivitasConfig.CIVITAS_CONFIDENCE_THRESHOLD}")

if __name__ == "__main__":
    # Test konfigurasi
    test_civitas_config()
    
    # Jalankan demo (uncomment untuk menjalankan)
    # example_civitas_detection()