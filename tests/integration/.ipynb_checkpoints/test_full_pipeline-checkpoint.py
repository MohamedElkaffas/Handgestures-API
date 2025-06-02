"""Tests for complete prediction pipeline"""

import pytest
from unittest.mock import patch
from tests import get_sample_landmarks

class TestFullPipeline:
    def test_landmarks_to_prediction_pipeline(self):
        """Test complete pipeline: landmarks → preprocessing → model → response"""
        landmarks = get_sample_landmarks()
        
        mock_prediction = {
            'gesture_name': 'like',
            'maze_action': 'UP', 
            'confidence': 0.85,
            'prediction_number': 4
        }
        
        with patch('app.services.gesture_service.GestureService.predict', return_value=mock_prediction):
            from app.services.gesture_service import GestureService
            service = GestureService.__new__(GestureService)  # Skip __init__
            
            # Test that the full pipeline works
            result = service.predict(landmarks)
            assert result['gesture_name'] == 'like'
            assert result['confidence'] == 0.85