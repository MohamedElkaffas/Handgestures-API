"""
Unit tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestHealthEndpoints:
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "Hand Gesture Maze Controller API" in data["service"]
        assert data["status"] == "running"
    
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

class TestPredictionEndpoints:
    def test_predict_valid_input(self):
        landmarks = [0.5] * 63
        response = client.post("/predict", json={"landmarks": landmarks})
        assert response.status_code == 200
        data = response.json()
        assert "gesture_name" in data
        assert "confidence" in data
        assert "maze_action" in data
    
    def test_predict_invalid_length(self):
        landmarks = [0.5] * 42  # Wrong length
        response = client.post("/predict", json={"landmarks": landmarks})
        assert response.status_code == 422  # Validation error
    
    def test_maze_control_endpoint(self):
        landmarks = [0.5] * 63
        response = client.post("/maze-control", json={"landmarks": landmarks})
        assert response.status_code == 200
        data = response.json()
        assert "action" in data
        assert "is_valid" in data
