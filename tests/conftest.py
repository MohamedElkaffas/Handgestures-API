"""
Pytest configuration and shared fixtures
"""
import pytest

# Only basic fixtures that don't import complex dependencies
@pytest.fixture
def simple_data():
    """Simple test data that doesn't require external libraries"""
    return {"test": "data", "number": 42}

@pytest.fixture
def sample_list():
    """Simple list for testing"""
    return [1, 2, 3, 4, 5]
