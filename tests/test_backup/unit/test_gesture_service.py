"""
Unit tests for GestureService class
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from app.services.gesture_service import GestureService

class TestGestureService:
    """Test cases for GestureService"""
    
    def test_init_success(self, mock_model, mock_label_encoder):
        """Test successful initialization"""
        with patch('joblib.load') as mock_load:
            mock_load.side_effect = [mock_model, mock_label_encoder]
            
            service = GestureService('fake_model.pkl', 'fake_encoder.pkl')
            
            assert service.model is not None
            assert service.label_encoder is not None
            assert service.health_check() is True
    
    def test_init_failure(self):
        """Test initialization failure"""
        with patch('joblib.load') as mock_load:
            mock_load.side_effect = Exception("File not found")
            
            service = GestureService('fake_model.pkl', 'fake_encoder.pkl')
            
            assert service.model is None
            assert service.label_encoder is None
            assert service.health_check() is False
    
    def test_preprocess_landmarks_correct_input(self, sample_landmarks):
        """Test preprocessing with correct 63 landmarks"""
        service = GestureService.__new__(GestureService)  # Create without __init__
        
        result = service.preprocess_landmarks(sample_landmarks)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (42,)  # 21 landmarks * 2 coordinates
        assert len(result) == 42
    
    def test_preprocess_landmarks_wrong_input_size(self):
        """Test preprocessing with wrong number of landmarks"""
        service = GestureService.__new__(GestureService)
        
        with pytest.raises(ValueError, match="Expected 63 landmarks, got 60"):
            service.preprocess_landmarks([0.5] * 60)
    
    def test_preprocess_landmarks_extracts_xy_only(self):
        """Test that preprocessing extracts only x,y coordinates"""
        service = GestureService.__new__(GestureService)
        
        # Create test data: [x1, y1, z1, x2, y2, z2, ...]
        test_landmarks = []
        for i in range(21):
            test_landmarks.extend([i * 0.1, i * 0.2, i * 0.3])  # x, y, z
        
        result = service.preprocess_landmarks(test_landmarks)
        
        # Should only contain x,y values, no z values
        expected = []
        for i in range(21):
            expected.extend([i * 0.1, i * 0.2])  # Only x, y
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_predict_success(self, mock_model, mock_label_encoder, sample_landmarks):
        """Test successful prediction"""
        with patch('joblib.load') as mock_load:
            mock_load.side_effect = [mock_model, mock_label_encoder]
            
            service = GestureService('fake_model.pkl', 'fake_encoder.pkl')
            result = service.predict(sample_landmarks)
            
            assert result['gesture_name'] == 'like'
            assert result['maze_action'] == 'UP'
            assert result['confidence'] == 0.85
            assert result['prediction_number'] == 4
    
    def test_predict_model_not_loaded(self, sample_landmarks):
        """Test prediction when model is not loaded"""
        service = GestureService.__new__(GestureService)
        service.model = None
        service.label_encoder = None
        
        with pytest.raises(RuntimeError, match="Model not loaded properly"):
            service.predict(sample_landmarks)
    
    def test_predict_preprocessing_error(self, mock_model, mock_label_encoder):
        """Test prediction with preprocessing error"""
        with patch('joblib.load') as mock_load:
            mock_load.side_effect = [mock_model, mock_label_encoder]
            
            service = GestureService('fake_model.pkl', 'fake_encoder.pkl')
            
            with pytest.raises(RuntimeError, match="Prediction failed"):
                service.predict([0.5] * 60)  # Wrong number of landmarks
    
    def test_gesture_to_action_mapping(self, mock_model, mock_label_encoder, sample_landmarks):
        """Test gesture to maze action mapping"""
        with patch('joblib.load') as mock_load:
            mock_load.side_effect = [mock_model, mock_label_encoder]
            
            service = GestureService('fake_model.pkl', 'fake_encoder.pkl')
            
            # Test different gesture predictions
            test_cases = [
                (0, 'call', 'ACTION'),
                (1, 'dislike', 'DOWN'), 
                (2, 'fist', 'WAIT'),
                (4, 'like', 'UP'),
                (7, 'one', 'LEFT'),
                (10, 'rock', 'RIGHT')
            ]
            
            for pred_num, gesture, expected_action in test_cases:
                mock_model.predict.return_value = np.array([pred_num])
                mock_label_encoder.inverse_transform.return_value = [gesture]
                
                result = service.predict(sample_landmarks)
                assert result['gesture_name'] == gesture
                assert result['maze_action'] == expected_action