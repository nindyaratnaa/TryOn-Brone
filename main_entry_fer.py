"""
Main Entry Point - FER Robot System
====================================
Entry point untuk menjalankan Publisher dan Subscriber bersamaan.

Jalankan: python main_entry_fer.py

Modules:
- FER Publisher (webcam emotion detection)
- FER Subscriber (robot face display)
- [Coming Soon] Gesture Recognition
- [Coming Soon] Voice Commands
"""
import subprocess
import sys
import os
import time
import signal
import platform

# ==================== CONFIGURATION ====================
class Config:
    # Auto-detect platform
    IS_JETSON = os.path.exists('/etc/nv_tegra_release')
    
    # Script paths (relative to this file)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    APP_DIR = os.path.join(BASE_DIR, 'app')
    
    # Scripts to run
    if IS_JETSON:
        PUBLISHER_SCRIPT = os.path.join(APP_DIR, 'publisher_jetson.py')
    else:
        PUBLISHER_SCRIPT = os.path.join(APP_DIR, 'publisher.py')
    
    SUBSCRIBER_SCRIPT = os.path.join(APP_DIR, 'subscriber.py')
    
    # Coming soon modules
    MODULES = {
        'fer_publisher': {
            'script': PUBLISHER_SCRIPT,
            'name': 'FER Publisher',
            'enabled': True
        },
        'fer_subscriber': {
            'script': SUBSCRIBER_SCRIPT,
            'name': 'FER Subscriber (Robot Face)',
            'enabled': True
        },
        'gesture': {
            'script': os.path.join(APP_DIR, 'gesture.py'),
            'name': 'Gesture Recognition',
            'enabled': False  # Coming soon
        },
        'voice': {
            'script': os.path.join(APP_DIR, 'voice_commands.py'),
            'name': 'Voice Commands',
            'enabled': False  # Coming soon
        }
    }

# ==================== PROCESS MANAGER ====================
class ProcessManager:
    """Manages multiple subprocesses"""
    
    def __init__(self):
        self.processes = {}
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n Shutdown signal received...")
        self.running = False
        self.stop_all()
    
    def start_module(self, module_id: str, script_path: str, name: str):
        """Start a module as subprocess"""
        if not os.path.exists(script_path):
            print(f" Script not found: {script_path}")
            return False
        
        try:
            print(f" Starting {name}...")
            process = subprocess.Popen(
                [sys.executable, script_path],
                cwd=os.path.dirname(script_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            self.processes[module_id] = {
                'process': process,
                'name': name,
                'script': script_path
            }
            print(f" {name} started (PID: {process.pid})")
            return True
        except Exception as e:
            print(f" Failed to start {name}: {e}")
            return False
    
    def stop_module(self, module_id: str):
        """Stop a specific module"""
        if module_id not in self.processes:
            return
        
        proc_info = self.processes[module_id]
        process = proc_info['process']
        name = proc_info['name']
        
        if process.poll() is None:  # Still running
            print(f" Stopping {name}...")
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()  
            print(f" {name} stopped")
    
    def stop_all(self):
        """Stop all running modules"""
        for module_id in list(self.processes.keys()):
            self.stop_module(module_id)
    
    def check_processes(self):
        """Check if any process has crashed"""
        for module_id, proc_info in list(self.processes.items()):
            process = proc_info['process']
            if process.poll() is not None:
                print(f" {proc_info['name']} has stopped (exit code: {process.returncode})")
                del self.processes[module_id]
    
    def is_any_running(self):
        """Check if any process is still running"""
        return any(p['process'].poll() is None for p in self.processes.values())

# ==================== MAIN ENTRY ====================
def main():
    print("=" * 60)
    print("BRONE - Robot Face Emotion System")
    print("=" * 60)
    print(f"Platform: {'Jetson' if Config.IS_JETSON else 'Intel/Laptop'}")
    print(f"Python: {sys.version.split()[0]}")
    print("=" * 60)
    
    # Show available modules
    print("\n Modules:")
    for module_id, module_info in Config.MODULES.items():
        status = " Enabled" if module_info['enabled'] else " Coming Soon"
        print(f"   - {module_info['name']}: {status}")
    print()
    
    # Initialize process manager
    manager = ProcessManager()
    
    # Start enabled modules
    for module_id, module_info in Config.MODULES.items():
        if module_info['enabled']:
            manager.start_module(
                module_id,
                module_info['script'],
                module_info['name']
            )
            time.sleep(1)  # Stagger startup
    
    if not manager.processes:
        print(" No modules started!")
        return
    
    print("\n" + "=" * 60)
    print(" All modules running!")
    print("   Press Ctrl+C to stop all modules")
    print("=" * 60 + "\n")
    
    # Main loop - monitor processes
    try:
        while manager.running and manager.is_any_running():
            manager.check_processes()
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_all()
        print("\n BRONE System stopped. Goodbye!")

if __name__ == "__main__":
    main()
