import subprocess
import time
import sys
import os
import signal

# ==================== CONFIGURATION ====================
# Daftar nama file script yang ingin dijalankan sekaligus
# Pastikan file-file ini berada di folder yang sama dengan main_entry.py
SCRIPTS_TO_RUN = [
    "jetson_run.py",           # Program 1: Publisher (Deteksi & MQTT)
    # "program_kedua.py"      # Program 2
]

# ==================== PROCESS MANAGER ====================
def main():
    print(f"--- JETSON MAIN LAUNCHER ---")
    print(f"Python Executable: {sys.executable}")
    print(f"Menjalankan {len(SCRIPTS_TO_RUN)} program secara paralel...")
    
    processes = []

    try:
        # 1. Jalankan semua script
        for script_name in SCRIPTS_TO_RUN:
            if not os.path.exists(script_name):
                print(f" Error: File {script_name} tidak ditemukan!")
                continue
                
            print(f" Memulai: {script_name}")
            
            # subprocess.Popen menjalankan script di background (non-blocking)
            # sys.executable memastikan kita menggunakan python yang sama (venv/system)
            p = subprocess.Popen([sys.executable, script_name])
            processes.append((script_name, p))
            
            #  jeda agar tidak rebutan resource CPU saat start
            time.sleep(2) 

        print("\n Semua sistem berjalan!")
        print("Tekan [Ctrl + C] di sini untuk mematikan SEMUA program.\n")

        # 2. Monitoring Loop (Watchdog)
        # Loop ini menjaga main.py tetap hidup dan memantau anak-anaknya
        while True:
            for name, p in processes:
                # poll() mengembalikan None jika proses masih hidup
                if p.poll() is not None:
                    print(f" Peringatan: {name} berhenti tiba-tiba! (Code: {p.returncode})")
                    # logika restart otomatis 
                    p = subprocess.Popen([sys.executable, name])
                    
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n Menerima perintah berhenti (Ctrl+C)...")
        print("Mematikan semua subsistem...")

    finally:
        # 3. Graceful Shutdown (Penting untuk melepas Kamera & MQTT)
        for name, p in processes:
            if p.poll() is None: # Jika masih hidup 
                print(f" Mematikan: {name}")
                p.terminate() # Kirim sinyal SIGTERM
                
                # Tunggu sebentar, jika bandel, paksa kill
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    p.kill()
        
        print(" Shutdown Selesai.")

if __name__ == "__main__":
    main()
