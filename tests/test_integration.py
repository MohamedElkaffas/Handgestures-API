"""
End-to-end integration tests
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestEndToEndFlow:
    def test_complete_prediction_flow(self):
        """Test complete flow from input to maze action"""
        landmarks = [0.5 + 0.1 * np.random.randn() for _ in range(63)]
        landmarks = [max(0, min(1, x)) for x in landmarks]  # Clamp to [0,1]
        
        # Test prediction endpoint
        response = client.post("/predict", json={"landmarks": landmarks})
        assert response.status_code == 200
        
        prediction = response.json()
        assert all(key in prediction for key in [
            "gesture_name", "maze_action", "confidence", "prediction_number"
        ])
        
        # Test maze control with same input
        response = client.post("/maze-control", json={"landmarks": landmarks})
        assert response.status_code == 200
        
        maze_control = response.json()
        assert maze_control["gesture"] == prediction["gesture_name"]
