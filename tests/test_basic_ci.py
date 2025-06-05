# tests/test_basic_ci.py
"""Simple tests"""

def test_imports():
    """Test that main modules can be imported"""
    try:
        from app.main import app
        from app.models import GestureInput, PredictionResponse
        from app.utils.config import get_settings
        assert app is not None
        assert GestureInput is not None
        assert PredictionResponse is not None
        assert get_settings is not None
    except ImportError as e:
        # If imports fail, at least we know what failed
        assert False, f"Import failed: {e}"

def test_settings():
    """Test settings configuration"""
    from app.utils.config import get_settings
    settings = get_settings()
    assert settings.MODEL_PATH is not None
    assert settings.ENCODER_PATH is not None
    assert settings.MIN_CONFIDENCE_THRESHOLD >= 0

def test_gesture_input_validation():
    """Test gesture input model"""
    from app.models import GestureInput
    import pytest
    
    # Valid input should work
    valid_landmarks = [0.5] * 63
    gesture_input = GestureInput(landmarks=valid_landmarks)
    assert len(gesture_input.landmarks) == 63
    
    # Invalid input should raise error
    with pytest.raises(Exception):
        invalid_landmarks = [0.5] * 62  # Wrong length
        GestureInput(landmarks=invalid_landmarks)
