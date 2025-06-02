"""
Integration tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from app.main import app

client = TestClient(app)

class TestAPIEndpoints:
    """Test cases for FastAPI endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API information"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Hand Gesture Maze Controller API"
        assert data["status"] == "running"
        assert "endpoints" in data
    
    def test_health_endpoint_healthy(self):
        """Test health endpoint when service is healthy"""
        with patch('app.main.gesture_service.health_check', return_value=True):
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_status"] is True
    
    def test_health_endpoint_unhealthy(self):
        """Test health endpoint when service is unhealthy"""
        with patch('app.main.gesture_service.health_check', return_value=False):
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["model_status"] is False
    
    def test_predict_endpoint_success(self, realistic_thumbs_up_landmarks):
        """Test successful prediction"""
        mock_prediction = {
            'gesture_name': 'like',
            'maze_action': 'UP',
            'confidence': 0.85,
            'prediction_number': 4
        }
        
        with patch('app.main.gesture_service.predict', return_value=mock_prediction):
            response = client.post(
                "/predict",
                json={"landmarks": realistic_thumbs_up_landmarks}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["gesture_name"] == "like"
            assert data["maze_action"] == "UP"
            assert data["confidence"] == 0.85
    
    def test_predict_endpoint_invalid_input(self):
        """Test prediction with invalid input"""
        response = client.post(
            "/predict",
            json={"landmarks": [0.5] * 60}  # Wrong number of landmarks
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_service_error(self, realistic_thumbs_up_landmarks):
        """Test prediction when service throws error"""
        with patch('app.main.gesture_service.predict', side_effect=Exception("Model error")):
            response = client.post(
                "/predict",
                json={"landmarks": realistic_thumbs_up_landmarks}
            )
            
            assert response.status_code == 500
            assert "Prediction error" in response.json()["detail"]
    
    def test_maze_control_endpoint_success(self, realistic_thumbs_up_landmarks):
        """Test successful maze control"""
        mock_prediction = {
            'gesture_name': 'like',
            'maze_action': 'UP',
            'confidence': 0.85,
            'prediction_number': 4
        }
        
        with patch('app.main.gesture_service.predict', return_value=mock_prediction):
            response = client.post(
                "/maze-control",
                json={"landmarks": realistic_thumbs_up_landmarks}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["action"] == "UP"
            assert data["gesture"] == "like"
            assert data["is_valid"] is True
    
    def test_maze_control_low_confidence(self, realistic_thumbs_up_landmarks):
        """Test maze control with low confidence prediction"""
        mock_prediction = {
            'gesture_name': 'like',
            'maze_action': 'UP',
            'confidence': 0.3,  # Below threshold
            'prediction_number': 4
        }
        
        with patch('app.main.gesture_service.predict', return_value=mock_prediction):
            response = client.post(
                "/maze-control",
                json={"landmarks": realistic_thumbs_up_landmarks}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["action"] == "WAIT"  # Should be WAIT due to low confidence
            assert data["is_valid"] is False