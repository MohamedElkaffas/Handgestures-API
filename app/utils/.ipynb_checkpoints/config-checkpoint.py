"""
Simple application configuration
No pydantic dependencies - just environment variables
"""

import os
from functools import lru_cache

class Settings:
    """Simple application settings"""
    
    def __init__(self):
        # Model paths
        self.MODEL_PATH = os.getenv("MODEL_PATH", "model/best_hand_gesture.pkl")
        self.ENCODER_PATH = os.getenv("ENCODER_PATH", "model/label_encoder.pkl")
        
        # API settings
        self.MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.7"))
        self.MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))
        
        # Monitoring
        self.ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        self.METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))

@lru_cache()
def get_settings():
    return Settings()