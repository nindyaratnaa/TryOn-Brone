import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
import os
import time
from collections import deque

# ==================== CONFIGURATION ====================
class Config:
    # Path Model (Pastikan file ini ada di folder models/)
    MODEL_PATH = 'models/fer_model_v1.2_fusion_colab.pth'
    
    # Haarcascade
    CASCADE_PATH = 'haarcascades/haarcascade_frontalface_default.xml'
    if not os.path.exists(CASCADE_PATH):
        # Fallback ke sistem jika file lokal tidak ada
        CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    # Settings Model
    NUM_CLASSES = 5 
    EMOTION_LABELS = ['Upset', 'Shocked', 'Happy', 'Sad', 'Neutral']
    INPUT_SIZE = 112

    # Civitas Detection Settings
    CIVITAS_LABELS = ['Non-Civitas UB', 'Civitas UB']
    UB_LOGO_TEMPLATES = {
        'colored': 'templates/ub_logo_colored.png',
        'bw': 'templates/ub_logo_bw.png'
    }
    UB_COLORS = {
        'navy': ([100, 100, 50], [120, 255, 200]),     # Navy blue almamater UB (lebih spesifik)
        'gold': ([10, 100, 100], [30, 255, 255]),      # Gold/yellow logo UB
        'dark_navy': ([105, 120, 30], [115, 255, 150]), # Dark navy almamater
        'light_navy': ([95, 80, 80], [125, 255, 220])   # Light navy untuk variasi pencahayaan
    }
    
    # Settings UI & Logika
    WINDOW_SIZE = 30            # Buffer 30 frame (~1 detik pada 30FPS) untuk smoothing stabil
    CONFIDENCE_THRESHOLD = 0.50 
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== MODEL DEFINITION ====================
class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes=5, architecture='resnet34'):
        super(EmotionRecognitionModel, self).__init__()
        self.backbone = models.resnet34(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ==================== UTILS (SMOOTHING) ====================
class TemporalAveraging:
    def __init__(self, window_size=15, confidence_threshold=0.5):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.buffer = deque(maxlen=window_size)
    
    def add_prediction(self, probabilities):
        self.buffer.append(probabilities)
    
    def get_averaged_emotion(self):
        # Butuh minimal data untuk mulai merata-rata
        if len(self.buffer) < 2: 
            return "Collecting...", 0.0
        
        avg_probs = np.mean(self.buffer, axis=0)
        idx = np.argmax(avg_probs)
        conf = np.max(avg_probs)
        
        if conf >= self.confidence_threshold:
            return Config.EMOTION_LABELS[idx], conf
        else:
            return "UNCERTAIN", conf
            
    def reset(self):
        self.buffer.clear()


class CivitasTemporalAveraging:
    def __init__(self, window_size=20, confidence_threshold=0.6):  # Threshold dinaikkan
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.buffer = deque(maxlen=window_size)
    
    def add_prediction(self, civitas_score, is_civitas):
        # Simpan sebagai [score, is_civitas_binary]
        self.buffer.append([civitas_score, 1.0 if is_civitas else 0.0])
    
    def get_averaged_civitas(self):
        if len(self.buffer) < 3:
            return "Detecting...", 0.0
        
        # Rata-rata score dan status
        scores = [item[0] for item in self.buffer]
        statuses = [item[1] for item in self.buffer]
        
        avg_score = np.mean(scores)
        avg_status = np.mean(statuses)  # 0.0-1.0
        
        # Tentukan status berdasarkan rata-rata (DIPERKETAT)
        if avg_status >= 0.6 and avg_score >= self.confidence_threshold:  # 60% frame positif
            return "Civitas UB", avg_score
        elif avg_status >= 0.4:  # 40% frame positif
            return "Uncertain", avg_score
        else:
            return "Non-Civitas UB", avg_score
    
    def reset(self):
        self.buffer.clear()


# ==================== CIVITAS DETECTION ====================
class CivitasDetector:
    def __init__(self):
        # Load multiple template logo UB
        self.ub_logo_templates = {}
        
        for template_type, path in Config.UB_LOGO_TEMPLATES.items():
            if os.path.exists(path):
                template = cv2.imread(path, 0)
                self.ub_logo_templates[template_type] = cv2.resize(template, (50, 50))
                print(f"✓ Loaded UB logo template: {template_type}")
    
    def detect_ub_logo(self, chest_roi):
        """Deteksi logo UB menggunakan multiple template matching dan deteksi warna"""
        if chest_roi.shape[0] < 50 or chest_roi.shape[1] < 50:
            return False, 0.0, None
            
        gray_chest = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2GRAY)
        hsv_chest = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2HSV)
        
        logo_score = 0.0
        logo_location = None
        
        # 1. Template matching dengan threshold ketat
        if self.ub_logo_templates:
            max_template_score = 0.0
            best_location = None
            
            for template_type, template in self.ub_logo_templates.items():
                result = cv2.matchTemplate(gray_chest, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > max_template_score:
                    max_template_score = max_val
                    best_location = (max_loc[0], max_loc[1], template.shape[1], template.shape[0])
            
            # Template matching threshold lebih rendah untuk area besar
            if max_template_score > 0.6:  # Threshold lebih rendah (dari 0.7)
                logo_score += max_template_score * 0.8  # Weight tinggi
                logo_location = best_location
        
        # 2. Deteksi kombinasi warna UB (navy + gold)
        navy_lower, navy_upper = Config.UB_COLORS['navy']
        gold_lower, gold_upper = Config.UB_COLORS['gold']
        
        navy_mask = cv2.inRange(hsv_chest, np.array(navy_lower), np.array(navy_upper))
        gold_mask = cv2.inRange(hsv_chest, np.array(gold_lower), np.array(gold_upper))
        
        navy_ratio = np.sum(navy_mask > 0) / (chest_roi.shape[0] * chest_roi.shape[1])
        gold_ratio = np.sum(gold_mask > 0) / (chest_roi.shape[0] * chest_roi.shape[1])
        
        # kombinasi navy + gold untuk logo UB 
        color_score = 0.0
        if navy_ratio > 0.04 and gold_ratio > 0.02: 
            # Cek apakah warna navy dan gold berdekatan (karakteristik logo UB)
            combined_mask = cv2.bitwise_or(navy_mask, gold_mask)
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Cari contour terbesar yang mungkin logo
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 150:  
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    aspect_ratio = w / h
                    
                    if 0.7 <= aspect_ratio <= 1.4:
                        color_score = min(navy_ratio * 4 + gold_ratio * 5, 0.6)  # Weight lebih tinggi
                        if logo_location is None:
                            logo_location = (x, y, w, h)
        
        logo_score += color_score
        
        # 3. Deteksi pola geometris khusus UB (lingkaran dengan teks)
        circles = cv2.HoughCircles(gray_chest, cv2.HOUGH_GRADIENT, 1, 30,
                                 param1=50, param2=30, minRadius=15, maxRadius=50)
        
        if circles is not None and len(circles[0]) > 0:
            for circle in circles[0]:
                cx, cy, radius = int(circle[0]), int(circle[1]), int(circle[2])
                
                mask = np.zeros(gray_chest.shape, dtype=np.uint8)
                cv2.circle(mask, (cx, cy), radius, 255, -1)
                
                masked_hsv = cv2.bitwise_and(hsv_chest, hsv_chest, mask=mask)
                circle_navy = cv2.inRange(masked_hsv, np.array(navy_lower), np.array(navy_upper))
                circle_gold = cv2.inRange(masked_hsv, np.array(gold_lower), np.array(gold_upper))
                
                circle_navy_ratio = np.sum(circle_navy > 0) / (np.pi * radius * radius)
                circle_gold_ratio = np.sum(circle_gold > 0) / (np.pi * radius * radius)
                
                if circle_navy_ratio > 0.05 and circle_gold_ratio > 0.02:  # Threshold lebih rendah
                    logo_score += 0.4  
                    if logo_location is None:
                        lx = max(0, cx - radius)
                        ly = max(0, cy - radius)
                        lw = min(radius * 2, chest_roi.shape[1] - lx)
                        lh = min(radius * 2, chest_roi.shape[0] - ly)
                        logo_location = (lx, ly, lw, lh)
                    break
        
        # Threshold untuk area besar
        is_ub_logo = (logo_score > 0.4) or (max_template_score > 0.6 if 'max_template_score' in locals() else False)
        
        return is_ub_logo, logo_score, logo_location  
    
    def detect_civitas_status(self, frame, x, y, w, h):
        """Deteksi status civitas berdasarkan almamater/jas + logo UB"""
        # Area dada untuk deteksi jas dan logo 
        chest_y = y + int(h * 0.6)
        chest_h = int(h * 2.2)    
        chest_x = max(0, x - int(w * 0.6)) 
        chest_w = int(w * 2.2)     
        
        # Boundary check
        chest_y = min(chest_y, frame.shape[0] - chest_h)
        chest_x = min(chest_x, frame.shape[1] - chest_w)
        chest_h = min(chest_h, frame.shape[0] - chest_y)
        chest_w = min(chest_w, frame.shape[1] - chest_x)
        
        if chest_h <= 0 or chest_w <= 0:
            return "Non-Civitas UB", 0.0, None
            
        chest_roi = frame[chest_y:chest_y+chest_h, chest_x:chest_x+chest_w]
        
        # 1. Deteksi warna jas (navy/formal)
        hsv = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2HSV)
        navy_lower, navy_upper = Config.UB_COLORS['navy']
        navy_mask = cv2.inRange(hsv, np.array(navy_lower), np.array(navy_upper))
        navy_ratio = np.sum(navy_mask > 0) / (chest_roi.shape[0] * chest_roi.shape[1])
        
        # 2. Deteksi logo UB
        has_logo, logo_confidence, logo_location = self.detect_ub_logo(chest_roi)
        
        # 3. Deteksi warna almamater navy UB (fokus biru navy)
        dark_navy_lower, dark_navy_upper = Config.UB_COLORS['dark_navy']
        dark_navy_mask = cv2.inRange(hsv, np.array(dark_navy_lower), np.array(dark_navy_upper))
        dark_navy_ratio = np.sum(dark_navy_mask > 0) / (chest_roi.shape[0] * chest_roi.shape[1])
        
        # Deteksi light navy untuk variasi pencahayaan
        light_navy_lower, light_navy_upper = Config.UB_COLORS['light_navy']
        light_navy_mask = cv2.inRange(hsv, np.array(light_navy_lower), np.array(light_navy_upper))
        light_navy_ratio = np.sum(light_navy_mask > 0) / (chest_roi.shape[0] * chest_roi.shape[1])
        
        # Gabungan semua variasi navy (almamater UB)
        navy_total_ratio = max(navy_ratio, dark_navy_ratio, light_navy_ratio)
        
        # 4. Logika klasifikasi civitas (FOKUS NAVY + LOGO)
        civitas_score = 0.0
        
        # CIVITAS UB - Navy almamater + Logo UB
        if has_logo and logo_confidence > 0.5 and navy_total_ratio > 0.25:
            # Logo UB jelas + Navy almamater kuat
            civitas_score = 0.85 + (logo_confidence * 0.15)
            status = "Civitas UB"
        elif has_logo and logo_confidence > 0.6 and navy_total_ratio > 0.15:
            # Logo UB sangat jelas + Navy almamater sedang
            civitas_score = 0.75 + (logo_confidence * 0.15)
            status = "Civitas UB"
        elif has_logo and logo_confidence > 0.7:
            # Logo UB sangat jelas (dosen/staff tanpa almamater)
            civitas_score = 0.70
            status = "Civitas UB"
        
        # NON-CIVITAS UB
        else:
            # Tanpa logo UB yang jelas = Non-Civitas
            if navy_total_ratio > 0.30:
                civitas_score = 0.35  # Navy tapi bukan UB (mungkin seragam lain)
            elif navy_total_ratio > 0.15:
                civitas_score = 0.25  # Sedikit navy
            else:
                civitas_score = 0.10  # Bukan navy
            status = "Non-Civitas UB"
        
        return status, civitas_score, (chest_x, chest_y, chest_w, chest_h), logo_location

# ==================== MAIN LIVE SYSTEM ====================
class LiveSystem:
    def __init__(self):
        print(f"--- FER V2 SYSTEM (Dashboard UI) ---")
        print(f"Device: {Config.DEVICE}")
        
        # 1. Load Model
        self.model = EmotionRecognitionModel(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
        
        if os.path.exists(Config.MODEL_PATH):
            print(f"Memuat model dari: {Config.MODEL_PATH}")
            checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.eval()
            print(" Model siap!")
        else:
            print(f" ERROR: Model tidak ditemukan di {Config.MODEL_PATH}")
            exit()

        # 2. Setup Camera
        self.face_cascade = cv2.CascadeClassifier(Config.CASCADE_PATH)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.temporal_avg = TemporalAveraging(Config.WINDOW_SIZE, Config.CONFIDENCE_THRESHOLD)
        self.civitas_temporal_avg = CivitasTemporalAveraging(window_size=20, confidence_threshold=0.6)
        self.no_face_counter = 0
        
        # Variabel FPS
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.fps = 0
        
        # Civitas detector
        self.civitas_detector = CivitasDetector()

    def draw_static_ui(self, frame):
        """Menggambar elemen UI yang selalu ada (Header)"""
        height, width, _ = frame.shape
        
        # Header Background
        cv2.rectangle(frame, (width//2 - 120, 0), (width//2 + 120, 40), (20, 20, 20), -1)
        
        # Header Text
        cv2.putText(frame, "FER V2 ", (width//2 - 90, 28),  
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def draw_dashboard(self, frame, x, y, w, h, instant_emo, instant_conf, smooth_emo, smooth_conf, 
                      instant_civitas_info=None, smooth_civitas_info=None):
        # --- LOGIKA WARNA (UPDATED) ---
        white = (255, 255, 255)
        orange = (0, 165, 255) # BGR
        green = (0, 255, 0)
        red = (0, 0, 255)
        blue = (255, 0, 0)
        gold = (0, 215, 255)  # Gold color
        gray = (128, 128, 128)
        
        if smooth_emo in ["Neutral", "Happy"]:
            status_color = green
        elif smooth_emo in ["Upset", "Sad", "Shocked"]:
            status_color = red
        else:
            status_color = orange

        # --- 1. Info Panel (Diperbesar untuk civitas info) ---
        overlay = frame.copy()
        panel_w, panel_h = 500, 280  # Tinggi diperbesar
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        start_x = 25
        start_y = 40
        spacing = 28

        # Baris 1: Instant Emotion
        cv2.putText(frame, f"Instant: {instant_emo} ({instant_conf:.2f})", 
                    (start_x, start_y), font, scale, white, 1, cv2.LINE_AA)

        # Baris 2: Smoothed Emotion
        cv2.putText(frame, f"Smoothed: {smooth_emo} ({smooth_conf:.2f})", 
                    (start_x, start_y + spacing), font, scale, status_color, 2, cv2.LINE_AA)

        # Baris 3: Instant Civitas Status
        if instant_civitas_info:
            civitas_status, civitas_conf = instant_civitas_info
            civitas_color = gold if civitas_status == "Civitas UB" else gray
            cv2.putText(frame, f"Instant Civitas: {civitas_status} ({civitas_conf:.2f})", 
                        (start_x, start_y + spacing * 2), font, scale, civitas_color, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Instant Civitas: Detecting...", 
                        (start_x, start_y + spacing * 2), font, scale, white, 1, cv2.LINE_AA)
        
        # Baris 4: Smoothed Civitas Status
        if smooth_civitas_info:
            smooth_civitas_status, smooth_civitas_conf = smooth_civitas_info
            if smooth_civitas_status == "Civitas UB":
                smooth_civitas_color = gold
            elif smooth_civitas_status == "Uncertain":
                smooth_civitas_color = orange
            else:
                smooth_civitas_color = gray
            cv2.putText(frame, f"Smoothed Civitas: {smooth_civitas_status} ({smooth_civitas_conf:.2f})", 
                        (start_x, start_y + spacing * 3), font, scale, smooth_civitas_color, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Smoothed Civitas: Collecting...", 
                        (start_x, start_y + spacing * 3), font, scale, white, 1, cv2.LINE_AA)

        # Baris 5: Debug Info (Tambahan untuk troubleshooting)
        if instant_civitas_info:
            _, civitas_conf = instant_civitas_info
            cv2.putText(frame, f"Debug - Logo Conf: {civitas_conf:.2f}", 
                        (start_x, start_y + spacing * 4), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Baris 6: Buffer Capacity
        buffer_len = len(self.temporal_avg.buffer)
        buffer_pct = int((buffer_len / Config.WINDOW_SIZE) * 100)
        civitas_buffer_len = len(self.civitas_temporal_avg.buffer)
        cv2.putText(frame, f"Emotion Buffer: {buffer_len}/{Config.WINDOW_SIZE} ({buffer_pct}%)", 
                    (start_x, start_y + spacing * 5), font, scale, white, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Civitas Buffer: {civitas_buffer_len}/20", 
                    (start_x, start_y + spacing * 6), font, scale, white, 1, cv2.LINE_AA)

        # Baris 7: FPS Counter
        cv2.putText(frame, f"FPS: {self.fps:.1f}", 
                    (start_x, start_y + spacing * 7), font, scale, white, 1, cv2.LINE_AA)

        # --- 2. Face Box & Label ---
        cv2.rectangle(frame, (x, y), (x+w, y+h), status_color, 2)
        
        # Label di atas kepala (gunakan smoothed civitas)
        label_text = f"{smooth_emo}"
        if smooth_civitas_info and smooth_civitas_info[0] == "Civitas UB":
            label_text += " | UB"
        elif smooth_civitas_info and smooth_civitas_info[0] == "Uncertain":
            label_text += " | ?"
        
        (tw, th), _ = cv2.getTextSize(label_text, font, 0.7, 2)
        cv2.rectangle(frame, (x, y - 35), (x + tw + 10, y), status_color, -1)
        cv2.putText(frame, label_text, (x + 5, y - 10), font, 0.7, (0, 0, 0), 2)

        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("❌ Gagal membuka webcam.")
            return

        print("🎥 Webcam dimulai. Tekan 'q' untuk keluar.")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # --- Hitung FPS ---
            self.new_frame_time = time.time()
            diff = self.new_frame_time - self.prev_frame_time
            self.fps = 1 / diff if diff > 0 else 0
            self.prev_frame_time = self.new_frame_time
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            
            # Gambar Header
            frame = self.draw_static_ui(frame)

            if len(faces) > 0:
                self.no_face_counter = 0
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                face_roi = frame[y:y+h, x:x+w]
                
                try:
                    # 1. Konversi BGR ke RGB
                    face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    
                    # 2. Inference
                    input_tensor = self.transform(face_roi_rgb).unsqueeze(0).to(Config.DEVICE)
                    
                    with torch.no_grad():
                        output = self.model(input_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
                    
                    # 3. Dapatkan Data Instant
                    instant_idx = np.argmax(probs)
                    instant_emo = Config.EMOTION_LABELS[instant_idx]
                    instant_conf = np.max(probs)
                    
                    # 4. Update Buffer & Dapatkan Data Smoothed
                    self.temporal_avg.add_prediction(probs)
                    smooth_emo, smooth_conf = self.temporal_avg.get_averaged_emotion()
                    
                    # 5. Deteksi status civitas (instant)
                    instant_civitas_status, instant_civitas_conf, civitas_box, logo_box = self.civitas_detector.detect_civitas_status(frame, x, y, w, h)
                    
                    # 6. Update civitas temporal averaging
                    is_civitas = instant_civitas_status == "Civitas UB"
                    self.civitas_temporal_avg.add_prediction(instant_civitas_conf, is_civitas)
                    smooth_civitas_status, smooth_civitas_conf = self.civitas_temporal_avg.get_averaged_civitas()
                    
                    # 7. Gambar civitas detection box
                    if civitas_box:
                        cx, cy, cw, ch = civitas_box
                        civitas_color = (0, 215, 255) if smooth_civitas_status == "Civitas UB" else (128, 128, 128)
                        cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), civitas_color, 1)
                        cv2.putText(frame, "Chest Area", (cx, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, civitas_color, 1)
                    
                    # 8. Gambar logo UB detection box (DIPERBAIKI)
                    if logo_box and civitas_box:
                        cx, cy, cw, ch = civitas_box
                        lx, ly, lw, lh = logo_box
                        
                        # Konversi koordinat logo dari chest_roi ke frame koordinat
                        logo_x = cx + lx
                        logo_y = cy + ly
                        
                        # Boundary checking untuk memastikan box tidak keluar frame
                        logo_x = max(0, min(logo_x, frame.shape[1] - lw))
                        logo_y = max(0, min(logo_y, frame.shape[0] - lh))
                        logo_w = min(lw, frame.shape[1] - logo_x)
                        logo_h = min(lh, frame.shape[0] - logo_y)
                        
                        # Pastikan ukuran box valid
                        if logo_w > 0 and logo_h > 0:
                            # Warna box berdasarkan confidence
                            if instant_civitas_conf > 0.7:
                                logo_color = (0, 255, 0)  # Hijau - confidence tinggi
                            elif instant_civitas_conf > 0.5:
                                logo_color = (0, 215, 255)  # Gold - confidence sedang
                            else:
                                logo_color = (0, 165, 255)  # Orange - confidence rendah
                            
                            cv2.rectangle(frame, (logo_x, logo_y), (logo_x+logo_w, logo_y+logo_h), logo_color, 2)
                            
                            # Label dengan posisi yang aman
                            label_y = max(logo_y - 5, 15)
                            cv2.putText(frame, f"UB Logo ({instant_civitas_conf:.2f})", 
                                       (logo_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, logo_color, 1)
                    
                    # 9. Gambar dashboard dengan info civitas realtime
                    frame = self.draw_dashboard(frame, x, y, w, h, instant_emo, instant_conf, 
                                              smooth_emo, smooth_conf, 
                                              (instant_civitas_status, instant_civitas_conf),
                                              (smooth_civitas_status, smooth_civitas_conf))
                    
                except Exception as e:
                    print(f"Error: {e}")
            else:
                self.no_face_counter += 1
                if self.no_face_counter > 10: 
                    self.temporal_avg.reset()
                    self.civitas_temporal_avg.reset()
                
                # Tetap tampilkan FPS saat idle
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (25, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                cv2.putText(frame, "Mencari Wajah...", (25, 115), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

            cv2.imshow('FER V2 + Civitas Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = LiveSystem()
    app.run()