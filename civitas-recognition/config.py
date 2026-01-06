"""
Configuration untuk Civitas Recognition System
"""
import os

class CivitasConfig:
    # Civitas Detection Settings
    CIVITAS_LABELS = ['Non-Civitas UB', 'Civitas UB']
    
    # Path ke template logo UB
    UB_LOGO_TEMPLATES = {
        'colored': 'templates/ub_logo_colored.png',
        'bw': 'templates/ub_logo_bw.png'
    }
    
    # Definisi warna UB dalam HSV
    UB_COLORS = {
        'navy': ([100, 100, 50], [120, 255, 200]),     # Navy blue almamater UB
        'gold': ([10, 100, 100], [30, 255, 255]),      # Gold/yellow logo UB
        'dark_navy': ([105, 120, 30], [115, 255, 150]), # Dark navy almamater
        'light_navy': ([95, 80, 80], [125, 255, 220])   # Light navy variasi pencahayaan
    }
    
    # Threshold untuk deteksi
    LOGO_CONFIDENCE_THRESHOLD = 0.4
    NAVY_RATIO_THRESHOLD = 0.25
    CIVITAS_CONFIDENCE_THRESHOLD = 0.6
    
    # Temporal averaging settings
    CIVITAS_WINDOW_SIZE = 20