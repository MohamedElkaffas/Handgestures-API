"""
Pytest configuration and shared fixtures
"""
import pytest

@pytest.fixture
def simple_data():
    """Simple test data as place holder"""
    return {"test": "data", "number": 42}

@pytest.fixture
def sample_list():
    """Simple list for testing"""
    return [1, 2, 3, 4, 5]
