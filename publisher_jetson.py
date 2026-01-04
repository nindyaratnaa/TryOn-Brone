"""
FER Publisher - Jetson Version (ONNX)
=====================================
Optimized for NVIDIA Jetson using ONNX Runtime

Features:
- ONNX model inference (faster than PyTorch on Jetson)
- GStreamer pipeline for CSI camera (with USB fallback)
- CUDA ExecutionProvider for GPU acceleration
- Optimized preprocessing

Jalankan: python3 publisher_jetson.py

Requirements:
- onnxruntime-gpu (for Jetson)
- OpenCV with GStreamer support (included in JetPack)
- paho-mqtt
- numpy

Install on Jetson:
  pip3 install onnxruntime-gpu paho-mqtt
"""
import cv2
import numpy as np
import os
import time
import json
import paho.mqtt.client as mqtt
from collections import deque

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    print(" ONNX Runtime loaded")
except ImportError:
    ONNX_AVAILABLE = False
    print(" ONNX Runtime not found!")

# ==================== CONFIGURATION ====================
class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fer_resnet34_v1.2.onnx')
    
    # Haarcascade
    CASCADE_PATH = 'haarcascades/haarcascade_frontalface_default.xml'
    if not os.path.exists(CASCADE_PATH):
        CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    # Model Settings
    NUM_CLASSES = 5
    EMOTION_LABELS = ['Upset', 'Shocked', 'Happy', 'Sad', 'Neutral']
    INPUT_SIZE = 112

    # Temporal Averaging
    WINDOW_SIZE = 15
    CONFIDENCE_THRESHOLD = 0.50

    # MQTT Settings
    MQTT_BROKER = "localhost"
    MQTT_PORT = 1883
    MQTT_TOPIC = "robot/expression"
    
    # Jetson-specific
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # GStreamer pipeline for CSI camera
    GST_PIPELINE = (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={CAMERA_WIDTH}, height={CAMERA_HEIGHT}, "
        f"format=NV12, framerate={CAMERA_FPS}/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, width={CAMERA_WIDTH}, height={CAMERA_HEIGHT}, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=1"
    )
    
    # Normalization values (ImageNet)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ==================== TEMPORAL AVERAGING ====================
class TemporalAveraging:
    """Buffer predictions over multiple frames for stability"""
    
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

# ==================== CAMERA FACTORY ====================
def get_camera():
    """
    Try to open camera with priority:
    1. CSI Camera via GStreamer (nvarguscamerasrc)
    2. USB Camera fallback
    """
    # Try CSI Camera first
    print(" Trying CSI camera (GStreamer)...")
    cap = cv2.VideoCapture(Config.GST_PIPELINE, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print(" CSI Camera opened successfully")
        return cap, "CSI"
    
    # Fallback to USB Camera
    print(" CSI not available, trying USB camera...")
    for i in range(3):  # Try /dev/video0, 1, 2
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
            print(f" USB Camera opened at /dev/video{i}")
            return cap, "USB"
    
    return None, None

# ==================== SOFTMAX ====================
def softmax(x):
    """Compute softmax values"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ==================== PUBLISHER SYSTEM ====================
class FERPublisherJetson:
    """Main Face Emotion Recognition Publisher for Jetson (ONNX)"""
    
    def __init__(self):
        print("=" * 50)
        print(" FER Publisher - JETSON Edition (ONNX)")
        print("=" * 50)

        # 1. Setup MQTT
        self.setup_mqtt()

        # 2. Load ONNX Model
        self.load_model()

        # 3. Setup Camera & Preprocessing
        self.setup_camera()

        # 4. Initialize Temporal Averaging
        self.temporal_avg = TemporalAveraging(
            Config.WINDOW_SIZE,
            Config.CONFIDENCE_THRESHOLD
        )
        
        # Tracking
        self.no_face_counter = 0
        self.prev_frame_time = 0
        self.fps = 0
        self.frame_count = 0

    def setup_mqtt(self):
        """Initialize MQTT client"""
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            self.client.on_connect = self.on_mqtt_connect
            self.client.connect(Config.MQTT_BROKER, Config.MQTT_PORT, 60)
            self.client.loop_start()
            print(f" MQTT: Connecting to {Config.MQTT_BROKER}:{Config.MQTT_PORT}")
        except Exception as e:
            print(f" MQTT Error: {e}")
            self.client = None

    def on_mqtt_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f" MQTT Connected! Topic: {Config.MQTT_TOPIC}")
        else:
            print(f" MQTT Connection failed: {rc}")

    def load_model(self):
        """Load ONNX model with GPU provider"""
        if not ONNX_AVAILABLE:
            print(" ONNX Runtime not available!")
            exit(1)
        
        if not os.path.exists(Config.MODEL_PATH):
            print(f" Model not found: {Config.MODEL_PATH}")
            exit(1)
        
        print(f" Loading ONNX model: {Config.MODEL_PATH}")
        
        # Try CUDA provider first, fallback to CPU
        providers = []
        
        # Check available providers
        available = ort.get_available_providers()
        print(f"   Available providers: {available}")
        
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
            print("   Using: CUDA (GPU)")
        elif 'TensorrtExecutionProvider' in available:
            providers.append('TensorrtExecutionProvider')
            print("   Using: TensorRT (GPU)")
        else:
            providers.append('CPUExecutionProvider')
            print("   Using: CPU (fallback)")
        
        # Create session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            Config.MODEL_PATH,
            sess_options=session_options,
            providers=providers
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f" Model loaded successfully")
        print(f"   Input: {self.input_name}")
        print(f"   Output: {self.output_name}")

    def setup_camera(self):
        """Setup camera and face detector"""
        self.face_cascade = cv2.CascadeClassifier(Config.CASCADE_PATH)
        
        if self.face_cascade.empty():
            print(f" Haarcascade not found at: {Config.CASCADE_PATH}")
            print("   Trying alternative path...")
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            if self.face_cascade.empty():
                print(" Haarcascade still not found!")
                exit(1)
        
        print(" Face detector ready")

    def preprocess_face(self, face_roi):
        """Preprocess face ROI for ONNX model"""
        # Resize to model input size
        face_resized = cv2.resize(face_roi, (Config.INPUT_SIZE, Config.INPUT_SIZE))
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        face_normalized = face_rgb.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        face_normalized = (face_normalized - Config.MEAN) / Config.STD
        
        # Change to CHW format (channels first)
        face_chw = np.transpose(face_normalized, (2, 0, 1))
        
        # Add batch dimension: (1, 3, 112, 112)
        input_tensor = np.expand_dims(face_chw, axis=0).astype(np.float32)
        
        return input_tensor

    def infer(self, input_tensor):
        """Run ONNX inference"""
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )
        return outputs[0][0]  # Return first batch, first output

    def publish_emotion(self, emotion: str, confidence: float):
        """Publish emotion to MQTT"""
        if emotion in ["UNCERTAIN", "Collecting...", "Analyzing..."]:
            return
        
        if self.client is None:
            return
        
        payload = {
            "timestamp": time.time(),
            "emotion": emotion,
            "confidence": round(float(confidence), 2)
        }
        
        try:
            self.client.publish(Config.MQTT_TOPIC, json.dumps(payload))
        except Exception as e:
            print(f" Publish error: {e}")

    def draw_ui(self, frame, x, y, w, h, emotion, confidence):
        """Draw bounding box and label on frame"""
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{emotion} ({confidence:.2f})"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame

    def run(self):
        """Main loop"""
        cap, cam_type = get_camera()
        
        if cap is None:
            print(" Cannot open any camera!")
            return
        
        print("\n" + "=" * 50)
        print(" FER Publisher Running (Jetson ONNX)")
        print(f"   Camera: {cam_type}")
        print("   Press Q to quit")
        print("=" * 50 + "\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(" Frame capture failed, retrying...")
                    time.sleep(0.1)
                    continue

                self.frame_count += 1

                # Calculate FPS
                current_time = time.time()
                if (current_time - self.prev_frame_time) > 0:
                    self.fps = 1 / (current_time - self.prev_frame_time)
                self.prev_frame_time = current_time

                # Flip horizontally (mirror) - only for USB camera display
                if cam_type == "USB":
                    frame = cv2.flip(frame, 1)
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(60, 60)
                )

                if len(faces) > 0:
                    self.no_face_counter = 0
                    
                    # Get largest face
                    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                    face_roi = frame[y:y + h, x:x + w]

                    try:
                        # Preprocess
                        input_tensor = self.preprocess_face(face_roi)

                        # ONNX Inference
                        logits = self.infer(input_tensor)
                        probs = softmax(logits)

                        # Temporal averaging
                        self.temporal_avg.add_prediction(probs)
                        emotion, confidence = self.temporal_avg.get_averaged_emotion()

                        # Publish
                        self.publish_emotion(emotion, confidence)
                        
                        # Draw UI
                        self.draw_ui(frame, x, y, w, h, emotion, confidence)

                    except Exception as e:
                        print(f" Processing error: {e}")
                else:
                    self.no_face_counter += 1
                    if self.no_face_counter > 10:
                        self.temporal_avg.reset()

                # Draw status info
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(frame, "ONNX Runtime", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                
                mqtt_status = "MQTT: OK" if self.client else "MQTT: Offline"
                cv2.putText(frame, mqtt_status, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                # Show frame
                cv2.imshow('FER Publisher (Jetson ONNX)', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n Interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
            print(" Publisher stopped")

# ==================== MAIN ====================
if __name__ == "__main__":
    app = FERPublisherJetson()
    app.run()
