import cv2
import numpy as np
import onnxruntime as ort
import os
import time
from collections import deque

# ==================== CONFIGURATION ====================
class Config:
    # Path ke file ONNX yang sudah Anda copy ke Jetson
    MODEL_PATH = 'fer_resnet34_v1.2.onnx' 
    
    # Path Haarcascade
    CASCADE_PATH = 'haarcascades/haarcascade_frontalface_default.xml'
    if not os.path.exists(CASCADE_PATH):
        # Fallback ke sistem opencv
        CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    EMOTION_LABELS = ['Upset', 'Shocked', 'Happy', 'Sad', 'Neutral']
    INPUT_SIZE = 112
    
    # Civitas Detection Settings
    CIVITAS_LABELS = ['Non-Civitas UB', 'Civitas UB']
    UB_LOGO_TEMPLATES = {
        'colored': 'templates/ub_logo_colored.png',
        'bw': 'templates/ub_logo_bw.png'
    }
    UB_COLORS = {
        'navy': ([100, 100, 50], [120, 255, 200]),     # Navy blue almamater UB
        'gold': ([10, 100, 100], [30, 255, 255]),      # Gold/yellow logo UB
        'dark_navy': ([105, 120, 30], [115, 255, 150]), # Dark navy almamater
        'light_navy': ([95, 80, 80], [125, 255, 220])   # Light navy  variasi pencahayaan
    }
    
    # UI Settings
    WINDOW_SIZE = 30
    CONFIDENCE_THRESHOLD = 0.50 

# ==================== UTILS (PRE-PROCESSING) ====================
def softmax(x):
    """Fungsi Softmax menggunakan Numpy (pengganti torch.softmax)"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def preprocess_image(face_img):
    """
    Mengubah gambar OpenCV (BGR) menjadi format yang dimengerti ONNX (Normalized Tensor).
    Sama persis dengan transforms.Normalize di PyTorch.
    """
    # 1. Resize
    img = cv2.resize(face_img, (Config.INPUT_SIZE, Config.INPUT_SIZE))
    
    # 2. Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Normalize & Standardize
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    # 4. Transpose (H, W, C) -> (C, H, W)
    img = img.transpose(2, 0, 1)
    
    # 5. Add Batch Dimension -> (1, 3, 112, 112)
    img = np.expand_dims(img, axis=0)
    
    return img

# ==================== UTILS (SMOOTHING) ====================
class TemporalAveraging:
    def __init__(self, window_size=15, confidence_threshold=0.5):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.buffer = deque(maxlen=window_size)
    
    def add_prediction(self, probabilities):
        self.buffer.append(probabilities)
    
    def get_averaged_emotion(self):
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
    def __init__(self, window_size=20, confidence_threshold=0.6):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.buffer = deque(maxlen=window_size)
    
    def add_prediction(self, civitas_score, is_civitas):
        self.buffer.append([civitas_score, 1.0 if is_civitas else 0.0])
    
    def get_averaged_civitas(self):
        if len(self.buffer) < 3:
            return "Detecting...", 0.0
        
        scores = [item[0] for item in self.buffer]
        statuses = [item[1] for item in self.buffer]
        
        avg_score = np.mean(scores)
        avg_status = np.mean(statuses)
        
        if avg_status >= 0.6 and avg_score >= self.confidence_threshold:
            return "Civitas UB", avg_score
        elif avg_status >= 0.4:
            return "Uncertain", avg_score
        else:
            return "Non-Civitas UB", avg_score
    
    def reset(self):
        self.buffer.clear()

# ==================== CIVITAS DETECTION ====================
class CivitasDetector:
    def __init__(self):
        self.ub_logo_templates = {}
        
        for template_type, path in Config.UB_LOGO_TEMPLATES.items():
            if os.path.exists(path):
                template = cv2.imread(path, 0)
                self.ub_logo_templates[template_type] = cv2.resize(template, (50, 50))
                print(f"✓ Loaded UB logo template: {template_type}")
    
    def detect_ub_logo(self, chest_roi):
        if chest_roi.shape[0] < 50 or chest_roi.shape[1] < 50:
            return False, 0.0, None
            
        gray_chest = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2GRAY)
        hsv_chest = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2HSV)
        
        logo_score = 0.0
        logo_location = None
        
        # Template matching
        if self.ub_logo_templates:
            max_template_score = 0.0
            best_location = None
            
            for template_type, template in self.ub_logo_templates.items():
                result = cv2.matchTemplate(gray_chest, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > max_template_score:
                    max_template_score = max_val
                    best_location = (max_loc[0], max_loc[1], template.shape[1], template.shape[0])
            
            if max_template_score > 0.6:
                logo_score += max_template_score * 0.8
                logo_location = best_location
        
        # Color detection
        navy_lower, navy_upper = Config.UB_COLORS['navy']
        gold_lower, gold_upper = Config.UB_COLORS['gold']
        
        navy_mask = cv2.inRange(hsv_chest, np.array(navy_lower), np.array(navy_upper))
        gold_mask = cv2.inRange(hsv_chest, np.array(gold_lower), np.array(gold_upper))
        
        navy_ratio = np.sum(navy_mask > 0) / (chest_roi.shape[0] * chest_roi.shape[1])
        gold_ratio = np.sum(gold_mask > 0) / (chest_roi.shape[0] * chest_roi.shape[1])
        
        if navy_ratio > 0.04 and gold_ratio > 0.02:
            combined_mask = cv2.bitwise_or(navy_mask, gold_mask)
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 150:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    aspect_ratio = w / h
                    
                    if 0.7 <= aspect_ratio <= 1.4:
                        color_score = min(navy_ratio * 4 + gold_ratio * 5, 0.6)
                        logo_score += color_score
                        if logo_location is None:
                            logo_location = (x, y, w, h)
        
        is_ub_logo = (logo_score > 0.4) or (max_template_score > 0.6 if 'max_template_score' in locals() else False)
        
        return is_ub_logo, logo_score, logo_location
    
    def detect_civitas_status(self, frame, x, y, w, h):
        chest_y = y + int(h * 0.6)
        chest_h = int(h * 2.2)
        chest_x = max(0, x - int(w * 0.6))
        chest_w = int(w * 2.2)
        
        chest_y = min(chest_y, frame.shape[0] - chest_h)
        chest_x = min(chest_x, frame.shape[1] - chest_w)
        chest_h = min(chest_h, frame.shape[0] - chest_y)
        chest_w = min(chest_w, frame.shape[1] - chest_x)
        
        if chest_h <= 0 or chest_w <= 0:
            return "Non-Civitas UB", 0.0, None
            
        chest_roi = frame[chest_y:chest_y+chest_h, chest_x:chest_x+chest_w]
        
        # Deteksi warna jas
        hsv = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2HSV)
        navy_lower, navy_upper = Config.UB_COLORS['navy']
        navy_mask = cv2.inRange(hsv, np.array(navy_lower), np.array(navy_upper))
        navy_ratio = np.sum(navy_mask > 0) / (chest_roi.shape[0] * chest_roi.shape[1])
        
        # Deteksi logo UB
        has_logo, logo_confidence, logo_location = self.detect_ub_logo(chest_roi)
        
        # Deteksi almamater navy
        dark_navy_lower, dark_navy_upper = Config.UB_COLORS['dark_navy']
        dark_navy_mask = cv2.inRange(hsv, np.array(dark_navy_lower), np.array(dark_navy_upper))
        dark_navy_ratio = np.sum(dark_navy_mask > 0) / (chest_roi.shape[0] * chest_roi.shape[1])
        
        light_navy_lower, light_navy_upper = Config.UB_COLORS['light_navy']
        light_navy_mask = cv2.inRange(hsv, np.array(light_navy_lower), np.array(light_navy_upper))
        light_navy_ratio = np.sum(light_navy_mask > 0) / (chest_roi.shape[0] * chest_roi.shape[1])
        
        navy_total_ratio = max(navy_ratio, dark_navy_ratio, light_navy_ratio)
        
        # Logika klasifikasi civitas
        civitas_score = 0.0
        
        if has_logo and logo_confidence > 0.5 and navy_total_ratio > 0.25:
            civitas_score = 0.85 + (logo_confidence * 0.15)
            status = "Civitas UB"
        elif has_logo and logo_confidence > 0.6 and navy_total_ratio > 0.15:
            civitas_score = 0.75 + (logo_confidence * 0.15)
            status = "Civitas UB"
        elif has_logo and logo_confidence > 0.7:
            civitas_score = 0.70
            status = "Civitas UB"
        else:
            if navy_total_ratio > 0.30:
                civitas_score = 0.35
            elif navy_total_ratio > 0.15:
                civitas_score = 0.25
            else:
                civitas_score = 0.10
            status = "Non-Civitas UB"
        
        return status, civitas_score, (chest_x, chest_y, chest_w, chest_h), logo_location

# ==================== MAIN JETSON SYSTEM ====================
class JetsonSystem:
    def __init__(self):
        print(f"--- FER V2 SYSTEM (JETSON ONNX EDITION) ---")
        
        # 1. Load ONNX Model
        if not os.path.exists(Config.MODEL_PATH):
            print(f"❌ Error: File {Config.MODEL_PATH} tidak ditemukan!")
            exit()

        print("Memuat ONNX Runtime...")
        try:
            # Prioritaskan GPU (CUDA), fallback ke CPU jika gagal
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.ort_session = ort.InferenceSession(Config.MODEL_PATH, providers=providers)
            print(f"✅ Model dimuat dengan Provider: {self.ort_session.get_providers()[0]}")
        except Exception as e:
            print(f"❌ Error loading ONNX: {e}")
            exit()
            
        # Dapatkan nama input & output layer
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

        # 2. Setup Camera
        self.face_cascade = cv2.CascadeClassifier(Config.CASCADE_PATH)
        self.temporal_avg = TemporalAveraging(Config.WINDOW_SIZE, Config.CONFIDENCE_THRESHOLD)
        self.civitas_temporal_avg = CivitasTemporalAveraging(window_size=20, confidence_threshold=0.6)
        self.no_face_counter = 0
        
        # Civitas detector
        self.civitas_detector = CivitasDetector()
        
        # FPS Vars
        self.prev_frame_time = 0
        self.fps = 0

    def draw_static_ui(self, frame):
        height, width, _ = frame.shape
        cv2.rectangle(frame, (width//2 - 120, 0), (width//2 + 120, 40), (20, 20, 20), -1)
        cv2.putText(frame, "FER V2 Jetson + UB", (width//2 - 110, 28), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def draw_dashboard(self, frame, x, y, w, h, instant_emo, instant_conf, smooth_emo, smooth_conf, 
                      instant_civitas_info=None, smooth_civitas_info=None):
        white = (255, 255, 255)
        orange = (0, 165, 255)
        green = (0, 255, 0)
        red = (0, 0, 255)
        gold = (0, 215, 255)
        gray = (128, 128, 128)
        
        if smooth_emo in ["Neutral", "Happy"]: status_color = green
        elif smooth_emo in ["Upset", "Sad", "Shocked"]: status_color = red
        else: status_color = orange

        # Info Panel (diperbesar untuk civitas info)
        overlay = frame.copy()
        panel_w, panel_h = 500, 280
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        start_x = 25
        start_y = 40
        spacing = 28

        # Emotion info
        cv2.putText(frame, f"Instant: {instant_emo} ({instant_conf:.2f})", 
                    (start_x, start_y), font, scale, white, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Smoothed: {smooth_emo} ({smooth_conf:.2f})", 
                    (start_x, start_y + spacing), font, scale, status_color, 2, cv2.LINE_AA)

        # Civitas info
        if instant_civitas_info:
            civitas_status, civitas_conf = instant_civitas_info
            civitas_color = gold if civitas_status == "Civitas UB" else gray
            cv2.putText(frame, f"Instant Civitas: {civitas_status} ({civitas_conf:.2f})", 
                        (start_x, start_y + spacing * 2), font, scale, civitas_color, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Instant Civitas: Detecting...", 
                        (start_x, start_y + spacing * 2), font, scale, white, 1, cv2.LINE_AA)
        
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

        # Buffer info
        buffer_len = len(self.temporal_avg.buffer)
        buffer_pct = int((buffer_len / Config.WINDOW_SIZE) * 100)
        civitas_buffer_len = len(self.civitas_temporal_avg.buffer)
        cv2.putText(frame, f"Emotion Buffer: {buffer_len}/{Config.WINDOW_SIZE} ({buffer_pct}%)", 
                    (start_x, start_y + spacing * 5), font, scale, white, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Civitas Buffer: {civitas_buffer_len}/20", 
                    (start_x, start_y + spacing * 6), font, scale, white, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", 
                    (start_x, start_y + spacing * 7), font, scale, white, 1, cv2.LINE_AA)

        # Face Box & Label
        cv2.rectangle(frame, (x, y), (x+w, y+h), status_color, 2)
        
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
        # Di Jetson, index kamera kadang 0 atau 1 (CSI Camera vs USB Webcam)
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
            
            # FPS Calculation
            new_frame_time = time.time()
            diff = new_frame_time - self.prev_frame_time
            self.fps = 1 / diff if diff > 0 else 0
            self.prev_frame_time = new_frame_time
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            
            frame = self.draw_static_ui(frame)

            if len(faces) > 0:
                self.no_face_counter = 0
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                face_roi = frame[y:y+h, x:x+w]
                
                try:
                    # 1. Preprocess Manual (Numpy Only)
                    input_tensor = preprocess_image(face_roi)
                    
                    # 2. Inference ONNX
                    # Run session dan ambil output pertama
                    ort_inputs = {self.input_name: input_tensor}
                    logits = self.ort_session.run([self.output_name], ort_inputs)[0]
                    
                    # 3. Softmax
                    probs = softmax(logits)[0]
                    
                    # 4. Logic UI (Sama seperti sebelumnya)
                    instant_idx = np.argmax(probs)
                    instant_emo = Config.EMOTION_LABELS[instant_idx]
                    instant_conf = np.max(probs)
                    
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
                    
                    # 8. Gambar logo UB detection box
                    if logo_box and civitas_box:
                        cx, cy, cw, ch = civitas_box
                        lx, ly, lw, lh = logo_box
                        
                        logo_x = cx + lx
                        logo_y = cy + ly
                        
                        logo_x = max(0, min(logo_x, frame.shape[1] - lw))
                        logo_y = max(0, min(logo_y, frame.shape[0] - lh))
                        logo_w = min(lw, frame.shape[1] - logo_x)
                        logo_h = min(lh, frame.shape[0] - logo_y)
                        
                        if logo_w > 0 and logo_h > 0:
                            if instant_civitas_conf > 0.7:
                                logo_color = (0, 255, 0)
                            elif instant_civitas_conf > 0.5:
                                logo_color = (0, 215, 255)
                            else:
                                logo_color = (0, 165, 255)
                            
                            cv2.rectangle(frame, (logo_x, logo_y), (logo_x+logo_w, logo_y+logo_h), logo_color, 2)
                            
                            label_y = max(logo_y - 5, 15)
                            cv2.putText(frame, f"UB Logo ({instant_civitas_conf:.2f})", 
                                       (logo_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, logo_color, 1)
                    
                    # 9. Gambar dashboard dengan info civitas
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
                cv2.putText(frame, "Mencari Wajah...", (25, 115), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            cv2.imshow('FER Jetson + Civitas UB', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = JetsonSystem()
    app.run()
