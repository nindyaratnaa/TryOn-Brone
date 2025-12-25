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
    # Rumus: (pixel - mean) / std
    # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
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
        self.no_face_counter = 0
        
        # FPS Vars
        self.prev_frame_time = 0
        self.fps = 0

    def draw_static_ui(self, frame):
        height, width, _ = frame.shape
        cv2.rectangle(frame, (width//2 - 120, 0), (width//2 + 120, 40), (20, 20, 20), -1)
        cv2.putText(frame, "FER V1.2 Jetson", (width//2 - 90, 28), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def draw_dashboard(self, frame, x, y, w, h, instant_emo, instant_conf, smooth_emo, smooth_conf):
        white = (255, 255, 255)
        orange = (0, 165, 255)
        green = (0, 255, 0)
        red = (0, 0, 255)
        
        if smooth_emo in ["Neutral", "Happy"]: status_color = green
        elif smooth_emo in ["Upset", "Sad", "Shocked"]: status_color = red
        else: status_color = orange

        # Info Panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (430, 170), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Instant: {instant_emo} ({instant_conf:.2f})", (25, 45), font, 0.65, white, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Smoothed: {smooth_emo} ({smooth_conf:.2f})", (25, 80), font, 0.65, status_color, 2, cv2.LINE_AA)
        
        buf_len = len(self.temporal_avg.buffer)
        buf_pct = int((buf_len / Config.WINDOW_SIZE) * 100)
        cv2.putText(frame, f"Buffer: {buf_len}/{Config.WINDOW_SIZE} ({buf_pct}%)", (25, 115), font, 0.65, white, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (25, 150), font, 0.65, white, 1, cv2.LINE_AA)

        # Face Box
        cv2.rectangle(frame, (x, y), (x+w, y+h), status_color, 2)
        
        # Label
        label = f"{smooth_emo}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
        cv2.rectangle(frame, (x, y - 35), (x + tw + 10, y), status_color, -1)
        cv2.putText(frame, label, (x + 5, y - 10), font, 0.7, (0, 0, 0), 2)

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
                    
                    frame = self.draw_dashboard(frame, x, y, w, h, instant_emo, instant_conf, smooth_emo, smooth_conf)
                    
                except Exception as e:
                    print(f"Error: {e}")
            else:
                self.no_face_counter += 1
                if self.no_face_counter > 10: self.temporal_avg.reset()
                cv2.putText(frame, "Mencari Wajah...", (25, 115), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            cv2.imshow('FER Jetson', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = JetsonSystem()
    app.run()
