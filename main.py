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
        'navy': ([100, 50, 50], [130, 255, 255]),     # Navy blue
        'gold': ([15, 100, 100], [25, 255, 255])      # Gold/yellow
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
            return False, 0.0
            
        gray_chest = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2GRAY)
        hsv_chest = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2HSV)
        
        logo_score = 0.0
        
        # 1. Multiple template matching
        if self.ub_logo_templates:
            max_template_score = 0.0
            for template_type, template in self.ub_logo_templates.items():
                result = cv2.matchTemplate(gray_chest, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                max_template_score = max(max_template_score, max_val)
            
            logo_score += max_template_score * 0.6  # 60% weight untuk template matching
        
        # 2. Deteksi kombinasi warna navy + gold (untuk logo berwarna)
        navy_lower, navy_upper = Config.UB_COLORS['navy']
        gold_lower, gold_upper = Config.UB_COLORS['gold']
        
        navy_mask = cv2.inRange(hsv_chest, np.array(navy_lower), np.array(navy_upper))
        gold_mask = cv2.inRange(hsv_chest, np.array(gold_lower), np.array(gold_upper))
        
        navy_ratio = np.sum(navy_mask > 0) / (chest_roi.shape[0] * chest_roi.shape[1])
        gold_ratio = np.sum(gold_mask > 0) / (chest_roi.shape[0] * chest_roi.shape[1])
        
        # Kombinasi navy + gold menandakan logo UB berwarna
        color_score = 0.0
        if navy_ratio > 0.05 and gold_ratio > 0.02:
            color_score = min(navy_ratio * 2 + gold_ratio * 3, 0.4)
        
        logo_score += color_score
        
        # 3. Deteksi pola geometris (lingkaran/emblem)
        circles = cv2.HoughCircles(gray_chest, cv2.HOUGH_GRADIENT, 1, 30,
                                 param1=50, param2=30, minRadius=10, maxRadius=40)
        
        if circles is not None and len(circles[0]) > 0:
            logo_score += 0.2
        
        return logo_score > 0.3, logo_score
    
    def detect_civitas_status(self, frame, x, y, w, h):
        """Deteksi status civitas berdasarkan almamater/jas + logo UB"""
        # Area dada untuk deteksi jas dan logo
        chest_y = y + int(h * 0.7)
        chest_h = int(h * 1.8)
        chest_x = max(0, x - int(w * 0.4))
        chest_w = int(w * 1.8)
        
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
        has_logo, logo_confidence = self.detect_ub_logo(chest_roi)
        
        # 3. Logika klasifikasi civitas
        civitas_score = 0.0
        
        # Jika ada jas formal (navy) + logo UB = Civitas UB
        if navy_ratio > 0.15 and has_logo:
            civitas_score = 0.7 + (logo_confidence * 0.3)
            status = "Civitas UB"
        # Jika hanya ada logo UB tanpa jas formal = kemungkinan civitas
        elif has_logo and logo_confidence > 0.5:
            civitas_score = 0.6
            status = "Civitas UB"
        # Jika hanya jas tanpa logo = bukan civitas UB
        elif navy_ratio > 0.20:
            civitas_score = 0.3
            status = "Non-Civitas UB"
        else:
            civitas_score = 0.1
            status = "Non-Civitas UB"
        
        return status, civitas_score, (chest_x, chest_y, chest_w, chest_h)

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

    def draw_dashboard(self, frame, x, y, w, h, instant_emo, instant_conf, smooth_emo, smooth_conf, civitas_info=None):
        # --- LOGIKA WARNA (UPDATED) ---
        white = (255, 255, 255)
        orange = (0, 165, 255) # BGR
        green = (0, 255, 0)
        red = (0, 0, 255)
        blue = (255, 0, 0)
        gold = (0, 215, 255)  # Gold color
        
        if smooth_emo in ["Neutral", "Happy"]:
            status_color = green
        elif smooth_emo in ["Upset", "Sad", "Shocked"]:
            status_color = red
        else:
            status_color = orange

        # --- 1. Info Panel (Diperbesar untuk civitas info) ---
        overlay = frame.copy()
        panel_w, panel_h = 450, 200
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.65
        start_x = 25
        start_y = 45
        spacing = 30

        # Baris 1: Instant
        cv2.putText(frame, f"Instant: {instant_emo} ({instant_conf:.2f})", 
                    (start_x, start_y), font, scale, white, 1, cv2.LINE_AA)

        # Baris 2: Smoothed
        cv2.putText(frame, f"Smoothed: {smooth_emo} ({smooth_conf:.2f})", 
                    (start_x, start_y + spacing), font, scale, status_color, 2, cv2.LINE_AA)

        # Baris 3: Civitas Status
        if civitas_info:
            civitas_status, civitas_conf = civitas_info
            civitas_color = gold if civitas_status == "Civitas UB" else white
            cv2.putText(frame, f"Status: {civitas_status} ({civitas_conf:.2f})", 
                        (start_x, start_y + spacing * 2), font, scale, civitas_color, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Status: Detecting...", 
                        (start_x, start_y + spacing * 2), font, scale, white, 1, cv2.LINE_AA)

        # Baris 4: Buffer Capacity
        buffer_len = len(self.temporal_avg.buffer)
        buffer_pct = int((buffer_len / Config.WINDOW_SIZE) * 100)
        cv2.putText(frame, f"Buffer: {buffer_len}/{Config.WINDOW_SIZE} ({buffer_pct}%)", 
                    (start_x, start_y + spacing * 3), font, scale, white, 1, cv2.LINE_AA)

        # Baris 5: FPS Counter
        cv2.putText(frame, f"FPS: {self.fps:.1f}", 
                    (start_x, start_y + spacing * 4), font, scale, white, 1, cv2.LINE_AA)

        # --- 2. Face Box & Label ---
        cv2.rectangle(frame, (x, y), (x+w, y+h), status_color, 2)
        
        # Label di atas kepala
        label_text = f"{smooth_emo}"
        if civitas_info and civitas_info[0] == "Civitas UB":
            label_text += " | UB"
        
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
                    
                    # 5. Deteksi status civitas
                    civitas_status, civitas_conf, civitas_box = self.civitas_detector.detect_civitas_status(frame, x, y, w, h)
                    
                    # 6. Gambar civitas detection box (opsional)
                    if civitas_box:
                        cx, cy, cw, ch = civitas_box
                        civitas_color = (0, 215, 255) if civitas_status == "Civitas UB" else (128, 128, 128)
                        cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), civitas_color, 1)
                    
                    # 7. Gambar dashboard dengan info civitas
                    frame = self.draw_dashboard(frame, x, y, w, h, instant_emo, instant_conf, 
                                              smooth_emo, smooth_conf, (civitas_status, civitas_conf))
                    
                except Exception as e:
                    print(f"Error: {e}")
            else:
                self.no_face_counter += 1
                if self.no_face_counter > 10: 
                    self.temporal_avg.reset()
                
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
