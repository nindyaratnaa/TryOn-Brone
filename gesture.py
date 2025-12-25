import cv2
import mediapipe as mp
import time

# --- Inisialisasi MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,                # Deteksi 1 tangan saja agar lebih fokus
    min_detection_confidence=0.7,   # Akurasi minimal untuk mendeteksi tangan
    min_tracking_confidence=0.7
)

# --- Variabel Logika Gesture ---
prev_x = 0
direction = 0  # 0: Diam, 1: Kanan, -1: Kiri
change_count = 0
last_change_time = 0
WAVE_THRESHOLD = 5  # Jumlah pergantian arah untuk dianggap melambai
display_text = ""
text_timer = 0

# --- Buka Kamera ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Balik frame (mirror) agar gerakan natural, dan ubah ke RGB untuk MediaPipe
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Proses deteksi tangan
    results = hands.process(rgb_frame)

    height, width, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar kerangka tangan
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --- LOGIKA DETEKSI LAMBAI ---
            
            # Ambil koordinat ujung jari tengah (Landmark ID 12)
            # MediaPipe memberikan koordinat relatif (0.0 - 1.0), kita kali dengan lebar layar
            middle_finger_tip = hand_landmarks.landmark[12]
            wrist = hand_landmarks.landmark[0]
            
            current_x = middle_finger_tip.x * width
            current_y = middle_finger_tip.y * height
            wrist_y = wrist.y * height

            # Cek apakah tangan tegak (jari di atas pergelangan)
            # Ingat: di koordinat gambar, Y=0 ada di paling atas. Jadi jari < wrist berarti jari di atas.
            if current_y < wrist_y:
                
                # Deteksi gerakan Kiri/Kanan
                if abs(current_x - prev_x) > 20: # Threshold gerakan (sensitivitas)
                    if current_x > prev_x:
                        current_dir = 1 # Gerak Kanan
                    else:
                        current_dir = -1 # Gerak Kiri

                    # Jika arah berubah (misal dari kiri ke kanan)
                    if current_dir != direction:
                        change_count += 1
                        last_change_time = time.time()
                        direction = current_dir

                    prev_x = current_x
            
            # Reset hitungan jika diam terlalu lama (lebih dari 1 detik)
            if time.time() - last_change_time > 1.0:
                change_count = 0

            # Jika arah berubah bolak-balik cukup sering, anggap melambai
            if change_count >= WAVE_THRESHOLD:
                display_text = "Gesture: MELAMBAI (WAVE) 👋"
                text_timer = time.time()
                change_count = 0 # Reset agar tidak spamming

    # --- Tampilkan Text ---
    # Teks akan hilang otomatis setelah 2 detik
    if time.time() - text_timer < 2:
        cv2.rectangle(frame, (0, 0), (width, 80), (245, 117, 16), -1) # Background oranye
        cv2.putText(frame, display_text, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Gesture Recognition Sederhana', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()