"""
FastAPI main application for Hand Gesture Maze Controller API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# ────────────────────────────────────────────────────────────────────────────────
#  Enable CORS so that the front end (served at http://localhost:3000 or file://)
#  can successfully call this API (which is listening on http://localhost:8001).
# ────────────────────────────────────────────────────────────────────────────────
origins = [
    # If you're hosting your front end on a specific domain, list it here:
    "https://mohamedelkaffas.github.io/MLOPs-Final-Project/",
    "https://mohamedelkaffas.github.io/MLOPs-Final-Project",
    "https://mohamedelkaffas.github.io",
    # Local development
    "http://localhost:8000",
    "http://localhost:8090",
    "http://127.0.0.1:8000", 
    "http://127.0.0.1:8090",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
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

# ────────────────────────────────────────────────────────────────────────────────
# CORS PREFLIGHT HANDLERS - Added to fix OPTIONS request issues
# ────────────────────────────────────────────────────────────────────────────────

@app.options("/predict")
async def options_predict():
    """Handle CORS preflight requests for /predict endpoint"""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
        }
    )

@app.options("/maze-control")
async def options_maze_control():
    """Handle CORS preflight requests for /maze-control endpoint"""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
        }
    )

@app.options("/health")
async def options_health():
    """Handle CORS preflight requests for /health endpoint"""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
        }
    )

@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle all other OPTIONS requests (CORS preflight)"""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
        }
    )

@app.get("/")
async def root():
    """Root endpoint - API status"""
    return {
        "service": "Hand Gesture Maze Controller API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": ["/predict", "/maze-control", "/health", "/docs"],
        "cors_enabled": True,
        "origins_configured": len(origins)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = gesture_service.health_check()
    return {
        "status": "healthy" if model_status else "unhealthy",
        "model_status": model_status,
        "timestamp": monitoring_service.get_timestamp()
    }

# ────────────────────────────────────────────────────────────────────────────────
# CORS TEST ENDPOINT - Added for debugging CORS issues
# ────────────────────────────────────────────────────────────────────────────────

@app.get("/test-cors")
async def test_cors():
    """Simple endpoint to test CORS configuration"""
    return {
        "message": "CORS is working correctly",
        "origins_configured": origins,
        "timestamp": monitoring_service.get_timestamp(),
        "cors_test": "success"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_gesture(input_data: GestureInput):
    """Predict hand gesture from MediaPipe landmarks"""
    try:
        if len(input_data.landmarks) != 63:
            monitoring_service.increment_error("invalid_input_length")
            raise HTTPException(
                status_code=422, 
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
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        monitoring_service.increment_error("prediction_error")
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {str(e)}")

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
        raise HTTPException(status_code=500, detail=f"Maze control error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)