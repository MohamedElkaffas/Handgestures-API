"""
Unit tests for configuration
"""

import pytest
from unittest.mock import patch
import os
from app.utils.config import Settings, get_settings

class TestConfiguration:
    """Test cases for application configuration"""
    
    def test_default_settings(self):
        """Test default configuration values"""
        settings = Settings()
        
        assert settings.MODEL_PATH == "model/best_hand_gesture.pkl"
        assert settings.ENCODER_PATH == "model/label_encoder.pkl"
        assert settings.MIN_CONFIDENCE_THRESHOLD == 0.7
        assert settings.MAX_REQUESTS_PER_MINUTE == 60
        assert settings.ENABLE_METRICS is True
        assert settings.METRICS_PORT == 8000
    
    def test_environment_variable_override(self):
        """Test configuration override with environment variables"""
        with patch.dict(os.environ, {
            'MODEL_PATH': 'custom/model.pkl',
            'MIN_CONFIDENCE_THRESHOLD': '0.8'
        }):
            settings = Settings()
            
            assert settings.MODEL_PATH == "custom/model.pkl"
            assert settings.MIN_CONFIDENCE_THRESHOLD == 0.8
    
    def test_get_settings_singleton(self):
        """Test that get_settings returns the same instance"""
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2