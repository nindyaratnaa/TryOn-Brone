"""
Civitas Recognition Module
Sistem deteksi civitas Universitas Brawijaya
"""

from civitas_recognition.config import CivitasConfig
from civitas_recognition.detector import CivitasDetector
from civitas_recognition.temporal_averaging import CivitasTemporalAveraging
from civitas_recognition.ui_utils import CivitasUI

__all__ = [
    'CivitasConfig',
    'CivitasDetector', 
    'CivitasTemporalAveraging',
    'CivitasUI'
]

__version__ = '1.0.0'
__author__ = 'TryOn-Brone Team'