"""
Temporal Averaging untuk Civitas Recognition
"""
import numpy as np
from collections import deque
from civitas_recognition.config import CivitasConfig

class CivitasTemporalAveraging:
    def __init__(self, window_size=None, confidence_threshold=None):
        self.window_size = window_size or CivitasConfig.CIVITAS_WINDOW_SIZE
        self.confidence_threshold = confidence_threshold or CivitasConfig.CIVITAS_CONFIDENCE_THRESHOLD
        self.buffer = deque(maxlen=self.window_size)
    
    def add_prediction(self, civitas_score, is_civitas):
        """Tambahkan prediksi civitas ke buffer"""
        self.buffer.append([civitas_score, 1.0 if is_civitas else 0.0])
    
    def get_averaged_civitas(self):
        """Dapatkan hasil civitas yang sudah di-smooth"""
        if len(self.buffer) < 3:
            return "Detecting...", 0.0
        
        scores = [item[0] for item in self.buffer]
        statuses = [item[1] for item in self.buffer]
        
        avg_score = np.mean(scores)
        avg_status = np.mean(statuses)
        
        # Logika klasifikasi berdasarkan rata-rata
        if avg_status >= 0.6 and avg_score >= self.confidence_threshold:
            return "Civitas UB", avg_score
        elif avg_status >= 0.4:
            return "Uncertain", avg_score
        else:
            return "Non-Civitas UB", avg_score
    
    def reset(self):
        """Reset buffer"""
        self.buffer.clear()
    
    def get_buffer_info(self):
        """Dapatkan informasi buffer untuk debugging"""
        return {
            'length': len(self.buffer),
            'capacity': self.window_size,
            'percentage': int((len(self.buffer) / self.window_size) * 100)
        }