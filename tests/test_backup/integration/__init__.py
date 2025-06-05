"""
Integration tests package

Contains integration tests that test multiple components working together:
- test_api_endpoints.py: Tests for FastAPI endpoint integration

Integration tests use real components where possible and test the full
request/response cycle.
"""

# Import parent test utilities
from .. import TEST_CONFIG, get_sample_landmarks, validate_prediction_response

# Integration test specific utilities
def create_test_client():
    """
    Create FastAPI test client for integration tests
    
    Returns:
        TestClient instance configured for testing
    """
    from fastapi.testclient import TestClient
    import sys
    import os
    
    # Ensure app is importable
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.insert(0, project_root)
    
    from app.main import app
    return TestClient(app)

def make_prediction_request(client, landmarks=None, endpoint="/predict"):
    """
    Helper to make prediction requests during integration tests
    
    Args:
        client: TestClient instance
        landmarks: List of landmark values (uses default if None)
        endpoint: API endpoint to call
        
    Returns:
        Response object
    """
    if landmarks is None:
        landmarks = get_sample_landmarks("thumbs_up")
    
    return client.post(endpoint, json={"landmarks": landmarks})

def assert_valid_api_response(response, expected_status=200):
    """
    Assert that API response has valid structure
    
    Args:
        response: Response object from TestClient
        expected_status: Expected HTTP status code
    """
    assert response.status_code == expected_status
    
    if expected_status == 200:
        data = response.json()
        if "gesture_name" in data:  # Prediction response
            assert validate_prediction_response(data)

# Integration test configuration
INTEGRATION_TEST_CONFIG = {
    **TEST_CONFIG,
    "api_base_url": "http://testserver",
    "timeout_seconds": 10,
    "expected_endpoints": ["/", "/health", "/predict", "/maze-control", "/docs"]
}

__all__ = [
    'INTEGRATION_TEST_CONFIG',
    'create_test_client',
    'make_prediction_request', 
    'assert_valid_api_response',
    'get_sample_landmarks',
    'validate_prediction_response'
]