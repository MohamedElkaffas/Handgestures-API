"""
Application configuration management
"""

from pydantic import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Model paths
    MODEL_PATH: str = "model/best_hand_gesture.pkl"
    ENCODER_PATH: str = "model/label_encoder.pkl"
    
    # API settings
    MIN_CONFIDENCE_THRESHOLD: float = 0.7
    MAX_REQUESTS_PER_MINUTE: int = 60
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 8000
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
