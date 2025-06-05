"""
Unit tests package

Contains unit tests for individual components:
- test_config.py: Tests for configuration management
- test_gesture_service.py: Tests for GestureService class
- test_models.py: Tests for Pydantic models

Unit tests focus on testing individual functions/classes in isolation
using mocks for external dependencies.
"""

# Import parent test utilities
from .. import TEST_CONFIG, get_sample_landmarks, validate_prediction_response

# Unit test specific utilities
def create_mock_model(prediction_class=4, confidence=0.85):
    """
    Create a mock ML model for unit testing
    
    Args:
        prediction_class: Class number to predict (default: 4 = 'like')
        confidence: Confidence score (default: 0.85)
        
    Returns:
        Mock object with predict and predict_proba methods
    """
    from unittest.mock import Mock
    import numpy as np
    
    mock_model = Mock()
    mock_model.predict.return_value = np.array([prediction_class])
    
    # Create probability array with high confidence for predicted class
    proba = np.zeros(11)  # Assuming 11 gesture classes
    proba[prediction_class] = confidence
    proba[proba == 0] = (1 - confidence) / 10  # Distribute remaining probability
    
    mock_model.predict_proba.return_value = np.array([proba])
    return mock_model

def create_mock_label_encoder(gesture_classes=None):
    """
    Create a mock label encoder for unit testing
    
    Args:
        gesture_classes: List of gesture class names
        
    Returns:
        Mock label encoder with classes_ and inverse_transform
    """
    from unittest.mock import Mock
    
    if gesture_classes is None:
        gesture_classes = [
            'call', 'dislike', 'fist', 'four', 'like', 'mute', 
            'ok', 'one', 'palm', 'peace', 'rock'
        ]
    
    mock_encoder = Mock()
    mock_encoder.classes_ = gesture_classes
    mock_encoder.inverse_transform = Mock(return_value=['like'])  # Default
    return mock_encoder

# Unit test configuration
UNIT_TEST_CONFIG = {
    **TEST_CONFIG,
    "mock_gesture_classes": [
        'call', 'dislike', 'fist', 'four', 'like', 'mute',
        'ok', 'one', 'palm', 'peace', 'rock'
    ],
    "default_prediction_class": 4,  # 'like'
    "default_confidence": 0.85
}

__all__ = [
    'UNIT_TEST_CONFIG',
    'create_mock_model',
    'create_mock_label_encoder',
    'get_sample_landmarks',
    'validate_prediction_response'
]