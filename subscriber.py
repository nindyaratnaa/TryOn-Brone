import cv2
import numpy as np
import paho.mqtt.client as mqtt
import json
import time
import os

# ==================== CONFIGURATION ====================
class Config:
    MQTT_BROKER = "localhost" 
    MQTT_TOPIC = "robot/expression"
    
    # Resolusi Tampilan Robot
    SCREEN_W = 800
    SCREEN_H = 600
    
    # Path Gambar Aset (Buat folder 'assets' dan isi gambar)
    ASSETS = {
        "Happy": "assets/happy.png",
        "Sad": "assets/sad.png",
        "Upset": "assets/upset.png",     # Angry + Disgust
        "Shocked": "assets/shocked.png", # Fear + Surprise
        "Neutral": "assets/neutral.png",
        "Idle": "assets/neutral.png"        # Saat diam
    }

# ==================== RECEIVER SYSTEM ====================
class RobotFaceDisplay:
    def __init__(self):
        print("--- ROBOT FACE SUBSCRIBER ---")
        self.current_emotion = "Idle"
        self.last_update_time = time.time()
        self.images = {}
        
        # 1. Load Images
        self.load_assets()
        
        # 2. Setup MQTT
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        try:
            print(f"Connecting to Broker {Config.MQTT_BROKER}...")
            self.client.connect(Config.MQTT_BROKER, 1883, 60)
            self.client.loop_start()
        except Exception as e:
            print(f"❌ Gagal konek broker: {e}")
            print("Pastikan Mosquitto service berjalan!")

    def load_assets(self):
        print("Memuat aset gambar...")
        # Buat dummy image hitam jika file tidak ada
        dummy = np.zeros((Config.SCREEN_H, Config.SCREEN_W, 3), dtype=np.uint8)
        
        for emotion, path in Config.ASSETS.items():
            if os.path.exists(path):
                img = cv2.imread(path)
                img = cv2.resize(img, (Config.SCREEN_W, Config.SCREEN_H))
                self.images[emotion] = img
            else:
                # Fallback: Buat gambar dengan teks jika file PNG tidak ada
                img = dummy.copy()
                color = (0, 255, 0) if emotion == "Happy" else (0, 0, 255)
                cv2.putText(img, f"[{emotion}]", (100, Config.SCREEN_H//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
                self.images[emotion] = img

    def on_connect(self, client, userdata, flags, rc):
        print(f"✅ Terhubung ke MQTT (Topik: {Config.MQTT_TOPIC})")
        client.subscribe(Config.MQTT_TOPIC)

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            new_emotion = payload.get("emotion", "Neutral")
            
            # Update state
            self.current_emotion = new_emotion
            self.last_update_time = time.time()
            
        except Exception as e:
            print(f"Error parsing JSON: {e}")

    def run(self):
        window_name = "Robot Face Output"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print("Display berjalan... Tekan 'q' untuk keluar.")
        
        while True:
            # Logic: Jika tidak ada data selama 5 detik, kembali ke Idle
            if time.time() - self.last_update_time > 5.0:
                target_emotion = "Idle"
            else:
                target_emotion = self.current_emotion
            
            # Ambil gambar
            if target_emotion not in self.images:
                target_emotion = "Neutral"
                
            frame = self.images[target_emotion]
            
            # Tampilkan
            cv2.imshow(window_name, frame)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        self.client.loop_stop()

if __name__ == "__main__":
    app = RobotFaceDisplay()
    app.run()
