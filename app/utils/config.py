"""
Application configuration management
Compatible with both Pydantic v1 and v2
"""

from functools import lru_cache

try:
    # Try Pydantic v1 import
    from pydantic import BaseSettings
except ImportError:
    # Fallback to Pydantic v2 import
    from pydantic_settings import BaseSettings

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
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()
