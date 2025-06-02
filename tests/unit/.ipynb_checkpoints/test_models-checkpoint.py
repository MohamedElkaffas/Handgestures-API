"""
Unit tests for Pydantic models
"""

import pytest
from pydantic import ValidationError
from app.models import GestureInput, PredictionResponse, MazeControlResponse

class TestPydanticModels:
    """Test cases for Pydantic request/response models"""
    
    def test_gesture_input_valid(self):
        """Test valid GestureInput"""
        landmarks = [0.5] * 63
        gesture_input = GestureInput(landmarks=landmarks)
        
        assert len(gesture_input.landmarks) == 63
        assert all(isinstance(x, float) for x in gesture_input.landmarks)
    
    def test_gesture_input_invalid_size(self):
        """Test GestureInput with wrong number of landmarks"""
        with pytest.raises(ValidationError):
            GestureInput(landmarks=[0.5] * 60)
    
    def test_gesture_input_invalid_type(self):
        """Test GestureInput with invalid data types"""
        with pytest.raises(ValidationError):
            GestureInput(landmarks=["invalid"] * 63)
    
    def test_prediction_response_valid(self):
        """Test valid PredictionResponse"""
        response = PredictionResponse(
            gesture_name="like",
            maze_action="UP",
            confidence=0.85,
            prediction_number=4
        )
        
        assert response.gesture_name == "like"
        assert response.maze_action == "UP"
        assert response.confidence == 0.85
    
    def test_maze_control_response_valid(self):
        """Test valid MazeControlResponse"""
        response = MazeControlResponse(
            action="UP",
            gesture="like",
            confidence=0.85,
            is_valid=True,
            threshold=0.7
        )
        
        assert response.action == "UP"
        assert response.is_valid is True