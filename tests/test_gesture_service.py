"""
Unit tests for gesture recognition service
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, mock_open
from app.services.gesture_service import GestureService

class TestGestureService:
    @pytest.fixture
    def mock_service(self):
        with patch('app.services.gesture_service.joblib.load') as mock_joblib, \
             patch('app.services.gesture_service.pickle.load') as mock_pickle, \
             patch('builtins.open', mock_open(read_data='{"maze_controls": {"thumbs_up": "UP"}}')), \
             patch('app.services.gesture_service.json.load') as mock_json:
            
            mock_json.return_value = {"maze_controls": {"thumbs_up": "UP"}}
            
            service = GestureService("fake_model.pkl", "fake_encoder.pkl")
            service.model = Mock()
            service.label_encoder = Mock()
            return service
    
    def test_predict_success(self, mock_service):
        mock_service.model.predict.return_value = np.array([0])
        mock_service.model.predict_proba.return_value = np.array([[0.9, 0.1]])
        mock_service.label_encoder.inverse_transform.return_value = ["thumbs_up"]
        
        landmarks = [0.5] * 63
        result = mock_service.predict(landmarks)
        
        assert result["gesture_name"] == "thumbs_up"
        assert result["maze_action"] == "UP"
        assert result["confidence"] == 0.9
    
    def test_health_check_healthy(self, mock_service):
        assert mock_service.health_check() == True
