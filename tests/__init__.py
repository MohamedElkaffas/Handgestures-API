"""
Test package for Hand Gesture API

This package contains unit tests, integration tests, and test utilities
for the hand gesture recognition API.

Test Structure:
- unit/: Unit tests for individual components
- integration/: Integration tests for API endpoints
- conftest.py: Shared fixtures and pytest configuration

Usage:
    pytest                    # Run all tests
    pytest tests/unit/        # Run only unit tests  
    pytest tests/integration/ # Run only integration tests
    pytest --cov=app         # Run with coverage
"""

import sys
import os
from pathlib import Path

# Add the app directory to Python path for imports
project_root = Path(__file__).parent.parent
app_path = project_root / "app"
sys.path.insert(0, str(app_path))
sys.path.insert(0, str(project_root))

# Test configuration
TEST_CONFIG = {
    "model_path": "model/best_hand_gesture.pkl",
    "encoder_path": "model/label_encoder.pkl", 
    "confidence_threshold": 0.7,
    "test_landmarks_size": 63,
    "processed_landmarks_size": 42
}

# Common test utilities
def get_sample_landmarks(gesture_type="neutral"):
    """
    Generate sample landmarks for testing
    
    Args:
        gesture_type: Type of gesture ("neutral", "thumbs_up", "fist", etc.)
    
    Returns:
        List of 63 landmark values
    """
    if gesture_type == "thumbs_up":
        # Realistic thumbs up landmarks
        return [
            0.50, 0.85, 0.0,    # Wrist
            0.45, 0.75, 0.0, 0.42, 0.65, 0.0, 0.40, 0.50, 0.0, 0.38, 0.35, 0.0,  # Thumb up
            0.55, 0.70, 0.0, 0.58, 0.75, 0.0, 0.60, 0.78, 0.0, 0.62, 0.80, 0.0,  # Index folded
            0.60, 0.72, 0.0, 0.63, 0.77, 0.0, 0.65, 0.79, 0.0, 0.67, 0.81, 0.0,  # Middle folded
            0.64, 0.74, 0.0, 0.66, 0.78, 0.0, 0.68, 0.80, 0.0, 0.70, 0.82, 0.0,  # Ring folded
            0.67, 0.76, 0.0, 0.68, 0.79, 0.0, 0.69, 0.81, 0.0, 0.70, 0.83, 0.0   # Pinky folded
        ]
    elif gesture_type == "fist":
        # All fingers closed
        return [0.5, 0.8, 0.0] + [0.5 + i*0.01, 0.75 + i*0.005, 0.0 for i in range(20)]
    else:
        # Neutral/default landmarks
        return [0.5 + i*0.01 for i in range(63)]

def validate_prediction_response(response_data):
    """
    Validate structure of prediction response
    
    Args:
        response_data: Dictionary containing prediction response
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_fields = ['gesture_name', 'maze_action', 'confidence', 'prediction_number']
    
    if not all(field in response_data for field in required_fields):
        return False
        
    if not isinstance(response_data['confidence'], (int, float)):
        return False
        
    if not 0 <= response_data['confidence'] <= 1:
        return False
        
    return True

# Import commonly used test modules for convenience
try:
    import pytest
    import numpy as np
    from unittest.mock import Mock, patch, MagicMock
    from fastapi.testclient import TestClient
    
    # Make these available to all test files
    __all__ = [
        'TEST_CONFIG',
        'get_sample_landmarks', 
        'validate_prediction_response',
        'pytest',
        'np',
        'Mock', 
        'patch',
        'MagicMock',
        'TestClient'
    ]
    
except ImportError as e:
    # Handle missing test dependencies gracefully
    print(f"Warning: Some test dependencies not available: {e}")
    __all__ = ['TEST_CONFIG', 'get_sample_landmarks', 'validate_prediction_response']