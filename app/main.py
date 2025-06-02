"""
FastAPI main application for Hand Gesture Maze Controller API
"""

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn

from app.models import GestureInput, PredictionResponse, MazeControlResponse
from app.services.gesture_service import GestureService
from app.services.monitoring_service import MonitoringService
from app.utils.config import get_settings

# Initialize services
settings = get_settings()
gesture_service = GestureService(settings.MODEL_PATH, settings.ENCODER_PATH)
monitoring_service = MonitoringService()

# FastAPI app
app = FastAPI(
    title="Hand Gesture Maze Controller API",
    description="Production API for real-time hand gesture recognition and maze game control",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Prometheus instrumentation
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=[".*admin.*", "/metrics"],
)
instrumentator.instrument(app).expose(app, include_in_schema=False)

@app.get("/")
async def root():
    """Root endpoint - API status"""
    return {
        "service": "Hand Gesture Maze Controller API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": ["/predict", "/maze-control", "/health", "/docs"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = gesture_service.health_check()
    return {
        "status": "healthy" if model_status else "unhealthy",
        "model_loaded": model_status,
        "timestamp": monitoring_service.get_timestamp()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_gesture(input_data: GestureInput):
    """Predict hand gesture from MediaPipe landmarks"""
    try:
        if len(input_data.landmarks) != 63:
            monitoring_service.increment_error("invalid_input_length")
            raise HTTPException(
                status_code=400, 
                detail=f"Expected 63 landmarks, got {len(input_data.landmarks)}"
            )
        
        # Get prediction
        prediction = gesture_service.predict(input_data.landmarks)
        
        # Update monitoring metrics
        monitoring_service.increment_prediction(prediction["gesture_name"])
        monitoring_service.record_confidence(prediction["confidence"])
        monitoring_service.increment_data_quality("valid")
        
        return PredictionResponse(**prediction)
        
    except ValueError as e:
        monitoring_service.increment_error("preprocessing_error")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        monitoring_service.increment_error("prediction_error")
        raise HTTPException(status_code=500, detail="Internal prediction error")

@app.post("/maze-control", response_model=MazeControlResponse)
async def maze_control(input_data: GestureInput):
    """Specialized endpoint for maze game control"""
    try:
        prediction = gesture_service.predict(input_data.landmarks)
        
        # Maze-specific logic
        is_valid = prediction["confidence"] >= settings.MIN_CONFIDENCE_THRESHOLD
        action = prediction["maze_action"] if is_valid else "WAIT"
        
        # Update maze-specific metrics
        monitoring_service.increment_maze_action(action)
        
        return MazeControlResponse(
            action=action,
            gesture=prediction["gesture_name"],
            confidence=prediction["confidence"],
            is_valid=is_valid,
            threshold=settings.MIN_CONFIDENCE_THRESHOLD
        )
        
    except Exception as e:
        monitoring_service.increment_error("maze_control_error")
        raise HTTPException(status_code=500, detail="Maze control error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
