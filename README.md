# 🎯 Hand Gesture Recognition API

Production-ready FastAPI service for real-time hand gesture recognition using MediaPipe landmarks. Powers a gesture-controlled maze game with 98.8% accuracy.

[![API Status](https://img.shields.io/badge/API-Live-brightgreen)](https://agkckrhhrjhv.eu-central-1.clawcloudrun.com/docs)
[![Docker](https://img.shields.io/badge/Docker-Available-blue)](https://hub.docker.com/r/scorpio1317/mlops_maze)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-orange)](https://github.com/MohamedElkaffas/Handgestures-API/actions)

## 🚀 Live Demo

- **🎮 Game**: [Play Gesture-Controlled Maze](https://mohamedelkaffas.github.io/MLOPs-Final-Project/)
- **📚 API Docs**: [Interactive API Documentation](https://oyrnfmwvrjoh.eu-central-1.clawcloudrun.com/docs)
- **📊 Monitoring**: Prometheus + Grafana dashboards included

## ✨ Features

- **Real-time gesture recognition** with 98.8% accuracy (SVM model)
- **MediaPipe integration** for 21-landmark hand tracking
- **Production-ready API** with FastAPI and Pydantic validation
- **Docker containerization** with multi-stage builds
- **CI/CD pipeline** with GitHub Actions
- **Comprehensive monitoring** with Prometheus and Grafana
- **CORS support** for web frontend integration
- **Health checks** and error handling

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │───▶│   FastAPI API    │───▶│  SVM Model      │
│ (MediaPipe +    │    │   (Port 8000)    │    │ (98.8% accuracy)│
│  Maze Game)     │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   Monitoring     │
                       │ (Prometheus +    │
                       │   Grafana)       │
                       └──────────────────┘
```

### Data Flow Pipeline

1. **MediaPipe** captures 21 hand landmarks (60 FPS)
2. **Frontend** normalizes 63 coordinates (x,y,z) to 0-1 range
3. **API** validates input and preprocesses 63→42 features
4. **SVM Model** predicts gesture with confidence score
5. **Response** returns gesture name, maze action, and confidence

## 📊 Model Performance

| Model                | Accuracy | F1 Score | Status       |
| -------------------- | :------: | :------: | ------------ |
| **SVM (Production)** | **98.8%**| **0.988**| ✅ Active    |
| XGBoost              | 97.9%    | 0.979    | 🔄 Candidate |
| Logistic Regression  | 84.7%    | 0.846    | 📊 Baseline  |

*All models tracked with MLflow for experiment management and reproducibility.*

## 🛠️ Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.8+ (for local development)
- Git

### 1. Clone Repository

```bash
git clone https://github.com/MohamedElkaffas/Handgestures-API.git
cd Handgestures-API
```

### 2. Run with Docker Compose

```bash
# Start all services (API + Monitoring)
docker-compose up -d

# Check status
docker-compose ps
```

### 3. Access Services

- **API**: http://localhost:8001
- **Docs**: http://localhost:8001/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin123)

### 4. Test API

```bash
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "landmarks": [
      0.50, 0.85, 0.0,    
      0.45, 0.75, 0.0,    
      0.42, 0.65, 0.0,    
      0.40, 0.50, 0.0,    
      0.38, 0.35, 0.0,    
      0.55, 0.70, 0.0,    
      0.58, 0.60, 0.0,    
      0.60, 0.65, 0.0,    
      0.62, 0.70, 0.0,    
      0.60, 0.72, 0.0,    
      0.63, 0.62, 0.0,    
      0.65, 0.68, 0.0,    
      0.67, 0.73, 0.0,    
      0.64, 0.74, 0.0,    
      0.66, 0.64, 0.0,    
      0.68, 0.70, 0.0,    
      0.70, 0.75, 0.0,    
      0.67, 0.76, 0.0,    
      0.68, 0.66, 0.0,    
      0.69, 0.72, 0.0,    
      0.70, 0.77, 0.0     
    ]
  }'
```

## 📚 API Reference

### Core Endpoints

#### `POST /predict`
Predict hand gesture from MediaPipe landmarks.

**Request Body:**
```json
{
  "landmarks": [
    0.50, 0.85, 0.0,    
    0.45, 0.75, 0.0,    
    0.42, 0.65, 0.0,    
    0.40, 0.50, 0.0,    
    0.38, 0.35, 0.0,    
    0.55, 0.70, 0.0,    
    0.58, 0.60, 0.0,    
    0.60, 0.65, 0.0,    
    0.62, 0.70, 0.0,    
    0.60, 0.72, 0.0,    
    0.63, 0.62, 0.0,    
    0.65, 0.68, 0.0,    
    0.67, 0.73, 0.0,    
    0.64, 0.74, 0.0,    
    0.66, 0.64, 0.0,    
    0.68, 0.70, 0.0,    
    0.70, 0.75, 0.0,    
    0.67, 0.76, 0.0,    
    0.68, 0.66, 0.0,    
    0.69, 0.72, 0.0,    
    0.70, 0.77, 0.0     
  ]
}
```

**Response:**
```json
{
  "gesture_name": "like",
  "maze_action": "UP", 
  "confidence": 0.98,
  "prediction_number": 0
}
```

#### `POST /maze-control`
Specialized endpoint for maze game control with confidence thresholding.

#### `GET /health`
Service health check and model status.

#### `GET /metrics`
Prometheus metrics for monitoring.

### Supported Gestures

| Gesture | Maze Action | Description |
|---------|-------------|-------------|
| `like` | UP | Thumbs up |
| `dislike` | DOWN | Thumbs down |
| `one` | LEFT | Index finger pointing |
| `rock` | RIGHT | Rock/metal hand sign |

## ⚙️ Configuration

### Environment Variables

```yaml
# Model Configuration
MODEL_PATH: "model/best_hand_gesture.pkl"
ENCODER_PATH: "model/label_encoder.pkl"

# API Settings  
MIN_CONFIDENCE_THRESHOLD: "0.1"  # 10% minimum confidence
MAX_REQUESTS_PER_MINUTE: "60"

# Monitoring
ENABLE_METRICS: "true"
METRICS_PORT: "8000"
```

### Docker Compose Services

```yaml
services:
  hand_gesture_api:
    image: docker.io/scorpio1317/mlops_maze:latest
    ports: ["8001:8000"]
    environment:
      - MIN_CONFIDENCE_THRESHOLD=0.1

  prometheus:
    image: prom/prometheus:v2.45.0
    ports: ["9090:9090"]

  grafana: 
    image: grafana/grafana:10.0.0
    ports: ["3000:3000"]
```

## 📈 Monitoring & Observability

### Prometheus Metrics

- `gesture_predictions_total{gesture_name}` - Prediction frequency by gesture
- `prediction_confidence_score` - Confidence distribution histogram  
- `api_errors_total{error_type}` - Error tracking by type
- `maze_actions_total{action}` - Game action frequency

### Grafana Dashboards

Pre-configured dashboards for:
- **API Performance**: Request latency, throughput, error rates
- **Model Metrics**: Prediction confidence, gesture patterns
- **System Health**: Resource usage, uptime monitoring

## 🚀 Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production (ClawCloud)

The API is automatically deployed via GitHub Actions CI/CD:

1. **Push to main** triggers build
2. **Docker image** built and pushed to registry
3. **ClawCloud** automatically deploys latest version
4. **Health checks** ensure successful deployment

**Live Production API**: https://agkckrhhrjhv.eu-central-1.clawcloudrun.com

### CI/CD Pipeline

```yaml
# Automated workflow on push
- Build Docker image
- Run tests and validation
- Push to Docker Hub
- Deploy to ClawCloud
- Health check verification
```

## 🧪 Development

### Project Structure

```
app/
├── main.py              # FastAPI application
├── models.py            # Pydantic models
├── services/
│   ├── gesture_service.py    # ML inference logic
│   └── monitoring_service.py # Metrics collection
└── utils/
    ├── config.py        # Configuration management
    └── preprocessing.py # Data preprocessing

model/                   # Trained ML models
docker-compose.yml       # Multi-service setup
Dockerfile              # Container configuration
requirements.txt        # Python dependencies
```

### Adding New Gestures

1. **Update gesture mapping** in `gesture_service.py`:
   ```python
   self.gesture_to_action = {
       'like': 'UP',
       'dislike': 'DOWN',
       'new_gesture': 'NEW_ACTION',  # Add here
       # ... existing mappings
   }
   ```

2. **Retrain model** with new gesture data
3. **Update model artifacts** in `/model` directory
4. **Test API endpoints** with new gestures
5. **Deploy** via CI/CD pipeline

### Core Processing Logic

```python
def preprocess_landmarks(self, landmarks: List[float]) -> np.ndarray:
    """
    Preprocess landmarks: 63 → 42 features
    """
    # Reshape to (21, 3) array
    landmarks_array = np.array(landmarks).reshape(21, 3)
    
    # Keep only x,y coordinates (drop z)
    xy_coordinates = landmarks_array[:, :2]
    
    # Wrist-relative positioning
    wrist = xy_coordinates[0, :]
    rel_coords = xy_coordinates - wrist
    
    # Scale normalization
    mid_tip = rel_coords[11, :]
    scale = np.linalg.norm(mid_tip)
    if scale == 0: scale = 1.0
    
    # Normalize and flatten
    normalized = rel_coords / scale
    return normalized.flatten()  # 42 features
```
