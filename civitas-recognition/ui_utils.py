"""
UI Utilities untuk Civitas Recognition
"""
import cv2

class CivitasUI:
    @staticmethod
    def get_civitas_colors():
        """Dapatkan warna untuk UI civitas"""
        return {
            'gold': (0, 215, 255),      # Gold untuk Civitas UB
            'gray': (128, 128, 128),    # Gray untuk Non-Civitas
            'orange': (0, 165, 255),    # Orange untuk Uncertain
            'white': (255, 255, 255),   # White untuk teks
            'green': (0, 255, 0),       # Green untuk logo detection
            'blue': (255, 0, 0)         # Blue untuk chest area
        }
    
    @staticmethod
    def draw_civitas_info(frame, start_x, start_y, spacing, instant_civitas_info, smooth_civitas_info):
        """Gambar informasi civitas pada frame"""
        colors = CivitasUI.get_civitas_colors()
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        
        # Instant Civitas Status
        if instant_civitas_info:
            civitas_status, civitas_conf = instant_civitas_info
            civitas_color = colors['gold'] if civitas_status == "Civitas UB" else colors['gray']
            cv2.putText(frame, f"Instant Civitas: {civitas_status} ({civitas_conf:.2f})", 
                        (start_x, start_y), font, scale, civitas_color, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Instant Civitas: Detecting...", 
                        (start_x, start_y), font, scale, colors['white'], 1, cv2.LINE_AA)
        
        # Smoothed Civitas Status
        if smooth_civitas_info:
            smooth_civitas_status, smooth_civitas_conf = smooth_civitas_info
            if smooth_civitas_status == "Civitas UB":
                smooth_civitas_color = colors['gold']
            elif smooth_civitas_status == "Uncertain":
                smooth_civitas_color = colors['orange']
            else:
                smooth_civitas_color = colors['gray']
            cv2.putText(frame, f"Smoothed Civitas: {smooth_civitas_status} ({smooth_civitas_conf:.2f})", 
                        (start_x, start_y + spacing), font, scale, smooth_civitas_color, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Smoothed Civitas: Collecting...", 
                        (start_x, start_y + spacing), font, scale, colors['white'], 1, cv2.LINE_AA)
    
    @staticmethod
    def draw_civitas_detection_boxes(frame, civitas_box, logo_box, instant_civitas_conf, smooth_civitas_status):
        """Gambar kotak deteksi civitas dan logo"""
        colors = CivitasUI.get_civitas_colors()
        
        # Gambar chest area box
        if civitas_box:
            cx, cy, cw, ch = civitas_box
            civitas_color = colors['gold'] if smooth_civitas_status == "Civitas UB" else colors['gray']
            cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), civitas_color, 1)
            cv2.putText(frame, "Chest Area", (cx, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, civitas_color, 1)
        
        # Gambar logo detection box
        if logo_box and civitas_box:
            cx, cy, cw, ch = civitas_box
            lx, ly, lw, lh = logo_box
            
            logo_x = cx + lx
            logo_y = cy + ly
            
            # Pastikan koordinat dalam batas frame
            logo_x = max(0, min(logo_x, frame.shape[1] - lw))
            logo_y = max(0, min(logo_y, frame.shape[0] - lh))
            logo_w = min(lw, frame.shape[1] - logo_x)
            logo_h = min(lh, frame.shape[0] - logo_y)
            
            if logo_w > 0 and logo_h > 0:
                # Warna berdasarkan confidence
                if instant_civitas_conf > 0.7:
                    logo_color = colors['green']
                elif instant_civitas_conf > 0.5:
                    logo_color = colors['gold']
                else:
                    logo_color = colors['orange']
                
                cv2.rectangle(frame, (logo_x, logo_y), (logo_x+logo_w, logo_y+logo_h), logo_color, 2)
                
                label_y = max(logo_y - 5, 15)
                cv2.putText(frame, f"UB Logo ({instant_civitas_conf:.2f})", 
                           (logo_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, logo_color, 1)
    
    @staticmethod
    def add_civitas_label_to_face(label_text, smooth_civitas_info):
        """Tambahkan label civitas ke label wajah"""
        if smooth_civitas_info and smooth_civitas_info[0] == "Civitas UB":
            label_text += " | UB"
        elif smooth_civitas_info and smooth_civitas_info[0] == "Uncertain":
            label_text += " | ?"
        return label_text