"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import List

class GestureInput(BaseModel):
    """Input model for hand gesture landmarks"""
    landmarks: List[float] = Field(
        ..., 
        min_items=63, 
        max_items=63, 
        description="63 MediaPipe hand landmarks (21 points × 3 coordinates)"
    )
    
    @validator('landmarks')
    def validate_landmarks(cls, v):
        if len(v) != 63:
            raise ValueError(f"Expected exactly 63 landmarks, got {len(v)}")
        if any(x < 0 or x > 1 for x in v):
            raise ValueError("Landmark coordinates must be normalized between 0 and 1")
        return v

class PredictionResponse(BaseModel):
    """Response model for gesture prediction"""
    gesture_name: str = Field(..., description="Recognized gesture name")
    maze_action: str = Field(..., description="Corresponding maze game action")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence score")
    prediction_number: int = Field(..., description="Numeric prediction for debugging")

class MazeControlResponse(BaseModel):
    """Response model for maze game control"""
    action: str = Field(..., description="Maze action command")
    gesture: str = Field(..., description="Detected gesture")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    is_valid: bool = Field(..., description="Whether confidence meets threshold")
    threshold: float = Field(..., description="Minimum confidence threshold")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: str = Field(..., description="Timestamp of health check")