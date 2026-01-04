"""
FER Subscriber - Robot Face Display
Menampilkan wajah robot berdasarkan emosi yang diterima via MQTT.

Jalankan: python app/subscriber.py
"""
import pygame
import json
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paho.mqtt.client as mqtt
from app.emotions import get_module, brone_neutral

# ==================== CONFIGURATION ====================
class Config:
    MQTT_BROKER = "localhost"
    MQTT_PORT = 1883
    MQTT_TOPIC = "robot/expression"
    
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600
    FPS = 60
    
    # Timeout: jika tidak ada data > 5 detik, kembali ke Neutral
    TIMEOUT_SECONDS = 5.0

# ==================== SUBSCRIBER CLASS ====================
class RobotFaceSubscriber:
    def __init__(self):
        print("🤖 Robot Face Subscriber Starting...")
        
        # Initialize Pygame ONCE
        pygame.init()
        self.screen = pygame.display.set_mode((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
        pygame.display.set_caption("Robot Face - FER Display")
        self.clock = pygame.time.Clock()
        
        # Current emotion module
        self.current_module = brone_neutral
        self.current_emotion = "Neutral"
        self.last_update_time = time.time()
        
        # MQTT Setup
        self.mqtt_connected = False
        self.setup_mqtt()
        
        print(f" Pygame Window Created: {Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")
    
    def setup_mqtt(self):
        """Initialize MQTT connection"""
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            self.client.on_connect = self.on_connect
            self.client.on_message = self.on_message
            self.client.on_disconnect = self.on_disconnect
            
            self.client.connect(Config.MQTT_BROKER, Config.MQTT_PORT, 60)
            self.client.loop_start()
            print(f" Connecting to MQTT Broker: {Config.MQTT_BROKER}:{Config.MQTT_PORT}")
        except Exception as e:
            print(f" MQTT Connection Error: {e}")
            print("   Running in standalone mode (manual emotion changes disabled)")
    
    def on_connect(self, client, userdata, flags, rc, properties=None):
        """MQTT on_connect callback"""
        if rc == 0:
            client.subscribe(Config.MQTT_TOPIC)
            self.mqtt_connected = True
            print(f" MQTT Connected! Subscribed to: {Config.MQTT_TOPIC}")
        else:
            print(f" MQTT Connection failed with code: {rc}")
    
    def on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties=None):
        """MQTT on_disconnect callback"""
        self.mqtt_connected = False
        print(f" MQTT Disconnected: {reason_code}")
    
    def on_message(self, client, userdata, msg):
        """MQTT on_message callback - handle incoming emotion data"""
        try:
            payload = json.loads(msg.payload.decode())
            new_emotion = payload.get("emotion", "Neutral")
            confidence = payload.get("confidence", 0.0)
            
            self.last_update_time = time.time()
            
            # Only switch if emotion changed
            if new_emotion != self.current_emotion:
                self.switch_emotion(new_emotion)
                print(f"🎭 Emotion: {new_emotion} (confidence: {confidence:.2f})")
                
        except json.JSONDecodeError as e:
            print(f" JSON Parse Error: {e}")
        except Exception as e:
            print(f" Message Error: {e}")
    
    def switch_emotion(self, emotion_name: str):
        """Switch to a different emotion module with transition blink"""
        self.current_module = get_module(emotion_name)
        self.current_emotion = emotion_name
        # Trigger blink for smooth transition
        if hasattr(self.current_module, 'trigger_blink'):
            self.current_module.trigger_blink()
    
    def check_timeout(self):
        """Check if we should switch to Neutral due to timeout"""
        if time.time() - self.last_update_time > Config.TIMEOUT_SECONDS:
            if self.current_emotion not in ["Neutral", "Idle"]:
                print(" Timeout: Switching to Neutral")
                self.switch_emotion("Neutral")
                self.last_update_time = time.time()  # Reset to avoid spam
    
    def run(self):
        """Main loop"""
        running = True
        
        print("\n" + "=" * 50)
        print(" Robot Face Display Running!")
        print("   Press Q or close window to exit")
        print("=" * 50 + "\n")
        
        try:
            while running:
                # Event handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            running = False
                        # Manual testing keys
                        elif event.key == pygame.K_1:
                            self.switch_emotion("Happy")
                            self.last_update_time = time.time()
                        elif event.key == pygame.K_2:
                            self.switch_emotion("Sad")
                            self.last_update_time = time.time()
                        elif event.key == pygame.K_3:
                            self.switch_emotion("Shocked")
                            self.last_update_time = time.time()
                        elif event.key == pygame.K_4:
                            self.switch_emotion("Upset")
                            self.last_update_time = time.time()
                        elif event.key == pygame.K_5:
                            self.switch_emotion("Neutral")
                            self.last_update_time = time.time()
                
                # Check timeout
                self.check_timeout()
                
                # Draw current emotion
                self.current_module.draw(self.screen)
                
                # Update display
                pygame.display.flip()
                self.clock.tick(Config.FPS)
                
        except KeyboardInterrupt:
            print("\n Interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print(" Cleaning up...")
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except:
            pass
        pygame.quit()
        print(" Goodbye!")

# ==================== MAIN ====================
if __name__ == "__main__":
    app = RobotFaceSubscriber()
    app.run()
