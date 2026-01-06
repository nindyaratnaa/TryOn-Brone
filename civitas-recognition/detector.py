"""
Civitas UB Detection System
"""
import cv2
import numpy as np
import os
from civitas_recognition.config import CivitasConfig

class CivitasDetector:
    def __init__(self):
        self.ub_logo_templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load template logo UB"""
        for template_type, path in CivitasConfig.UB_LOGO_TEMPLATES.items():
            if os.path.exists(path):
                template = cv2.imread(path, 0)
                if template is not None:
                    self.ub_logo_templates[template_type] = cv2.resize(template, (50, 50))
                    print(f"✓ Loaded UB logo template: {template_type}")
            else:
                print(f"⚠ Template not found: {path}")
    
    def detect_ub_logo(self, chest_roi):
        """Deteksi logo UB pada ROI dada"""
        if chest_roi.shape[0] < 50 or chest_roi.shape[1] < 50:
            return False, 0.0, None
            
        gray_chest = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2GRAY)
        hsv_chest = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2HSV)
        
        logo_score = 0.0
        logo_location = None
        max_template_score = 0.0
        
        # Template matching
        if self.ub_logo_templates:
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
        
        # Color detection untuk logo UB (navy + gold)
        navy_lower, navy_upper = CivitasConfig.UB_COLORS['navy']
        gold_lower, gold_upper = CivitasConfig.UB_COLORS['gold']
        
        navy_mask = cv2.inRange(hsv_chest, np.array(navy_lower), np.array(navy_upper))
        gold_mask = cv2.inRange(hsv_chest, np.array(gold_lower), np.array(gold_upper))
        
        navy_ratio = np.sum(navy_mask > 0) / (chest_roi.shape[0] * chest_roi.shape[1])
        gold_ratio = np.sum(gold_mask > 0) / (chest_roi.shape[0] * chest_roi.shape[1])
        
        # Kombinasi navy dan gold untuk logo
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
        
        is_ub_logo = (logo_score > CivitasConfig.LOGO_CONFIDENCE_THRESHOLD) or (max_template_score > 0.6)
        
        return is_ub_logo, logo_score, logo_location
    
    def detect_civitas_status(self, frame, face_x, face_y, face_w, face_h):
        """Deteksi status civitas berdasarkan wajah yang terdeteksi"""
        # Definisi area dada berdasarkan posisi wajah
        chest_y = face_y + int(face_h * 0.6)
        chest_h = int(face_h * 2.2)
        chest_x = max(0, face_x - int(face_w * 0.6))
        chest_w = int(face_w * 2.2)
        
        # Pastikan ROI dalam batas frame
        chest_y = min(chest_y, frame.shape[0] - chest_h)
        chest_x = min(chest_x, frame.shape[1] - chest_w)
        chest_h = min(chest_h, frame.shape[0] - chest_y)
        chest_w = min(chest_w, frame.shape[1] - chest_x)
        
        if chest_h <= 0 or chest_w <= 0:
            return "Non-Civitas UB", 0.0, None, None
            
        chest_roi = frame[chest_y:chest_y+chest_h, chest_x:chest_x+chest_w]
        
        # 1. Deteksi logo UB
        has_logo, logo_confidence, logo_location = self.detect_ub_logo(chest_roi)
        
        # 2. Deteksi warna almamater navy
        hsv = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2HSV)
        
        # Navy standar
        navy_lower, navy_upper = CivitasConfig.UB_COLORS['navy']
        navy_mask = cv2.inRange(hsv, np.array(navy_lower), np.array(navy_upper))
        navy_ratio = np.sum(navy_mask > 0) / (chest_roi.shape[0] * chest_roi.shape[1])
        
        # Dark navy
        dark_navy_lower, dark_navy_upper = CivitasConfig.UB_COLORS['dark_navy']
        dark_navy_mask = cv2.inRange(hsv, np.array(dark_navy_lower), np.array(dark_navy_upper))
        dark_navy_ratio = np.sum(dark_navy_mask > 0) / (chest_roi.shape[0] * chest_roi.shape[1])
        
        # Light navy
        light_navy_lower, light_navy_upper = CivitasConfig.UB_COLORS['light_navy']
        light_navy_mask = cv2.inRange(hsv, np.array(light_navy_lower), np.array(light_navy_upper))
        light_navy_ratio = np.sum(light_navy_mask > 0) / (chest_roi.shape[0] * chest_roi.shape[1])
        
        # Total navy ratio
        navy_total_ratio = max(navy_ratio, dark_navy_ratio, light_navy_ratio)
        
        # 3. Logika klasifikasi civitas
        civitas_score = 0.0
        
        # CIVITAS UB - Navy almamater + Logo UB
        if has_logo and logo_confidence > 0.5 and navy_total_ratio > CivitasConfig.NAVY_RATIO_THRESHOLD:
            # Logo UB jelas + Navy almamater kuat
            civitas_score = 0.85 + (logo_confidence * 0.15)
            status = "Civitas UB"
        elif has_logo and logo_confidence > 0.6 and navy_total_ratio > 0.15:
            # Logo UB sangat jelas + Navy almamater sedang
            civitas_score = 0.75 + (logo_confidence * 0.15)
            status = "Civitas UB"
        elif has_logo and logo_confidence > 0.7:
            # Logo UB sangat jelas (dosen/staff tanpa almamater)
            civitas_score = 0.70
            status = "Civitas UB"
        else:
            # NON-CIVITAS UB
            if navy_total_ratio > 0.30:
                civitas_score = 0.35  # Navy tapi bukan UB
            elif navy_total_ratio > 0.15:
                civitas_score = 0.25  # Sedikit navy
            else:
                civitas_score = 0.10  # Bukan navy
            status = "Non-Civitas UB"
        
        return status, civitas_score, (chest_x, chest_y, chest_w, chest_h), logo_location