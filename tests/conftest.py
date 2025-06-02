"""
Pytest configuration and shared fixtures
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def sample_landmarks():
    """Sample hand landmarks for testing (63 values)"""
    return [0.5] * 63

@pytest.fixture
def realistic_thumbs_up_landmarks():
    """Realistic thumbs up gesture landmarks"""
    return [
        0.50, 0.85, 0.0,    # Wrist
        0.45, 0.75, 0.0,    # Thumb base 
        0.42, 0.65, 0.0,    # Thumb joint 1
        0.40, 0.50, 0.0,    # Thumb joint 2  
        0.38, 0.35, 0.0,    # Thumb tip (UP!)
        0.55, 0.70, 0.0,    # Index base
        0.58, 0.60, 0.0, 0.60, 0.65, 0.0, 0.62, 0.70, 0.0,
        0.60, 0.72, 0.0, 0.63, 0.62, 0.0, 0.65, 0.68, 0.0, 0.67, 0.73, 0.0,
        0.64, 0.74, 0.0, 0.66, 0.64, 0.0, 0.68, 0.70, 0.0, 0.70, 0.75, 0.0,
        0.67, 0.76, 0.0, 0.68, 0.66, 0.0, 0.69, 0.72, 0.0, 0.70, 0.77, 0.0
    ]

@pytest.fixture
def mock_model():
    """Mock machine learning model"""
    model = Mock()
    model.predict.return_value = np.array([4])  # Predict 'like' class
    model.predict_proba.return_value = np.array([[0.1, 0.05, 0.1, 0.05, 0.85, 0.1]])
    return model

@pytest.fixture
def mock_label_encoder():
    """Mock label encoder"""
    encoder = Mock()
    encoder.classes_ = ['call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'rock']
    encoder.inverse_transform.return_value = ['like']
    return encoder