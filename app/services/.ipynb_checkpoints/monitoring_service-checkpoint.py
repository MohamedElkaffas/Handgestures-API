"""
Custom monitoring metrics for gesture recognition API
"""

from prometheus_client import Counter, Histogram, Gauge, Info
from datetime import datetime

class MonitoringService:
    """Service for collecting custom application metrics"""
    
    def __init__(self):
        # MODEL-RELATED METRICS
        self.gesture_predictions = Counter(
            'gesture_predictions_total',
            'Total number of gesture predictions made',
            ['gesture_name']
        )
        
        self.prediction_confidence = Histogram(
            'prediction_confidence_score',
            'Distribution of model confidence scores',
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # DATA-RELATED METRICS
        self.input_data_quality = Counter(
            'input_data_quality_total',
            'Quality indicators for input data',
            ['quality_type']  # valid, invalid_length, invalid_range, malformed
        )
        
        # SERVER-RELATED METRICS  
        self.api_errors = Counter(
            'api_errors_total',
            'Total number of API errors by type',
            ['error_type']
        )
        
        self.maze_actions = Counter(
            'maze_actions_total',
            'Total maze game actions triggered',
            ['action']  # UP, DOWN, LEFT, RIGHT, STOP, WAIT
        )
        
        # System info
        self.app_info = Info(
            'hand_gesture_api_info',
            'Information about the hand gesture API'
        )
        self.app_info.info({
            'version': '1.0.0',
            'model_type': 'gesture_recognition',
            'use_case': 'maze_game_controller'
        })
    
    def increment_prediction(self, gesture_name: str):
        """Record a gesture prediction (MODEL METRIC)"""
        self.gesture_predictions.labels(gesture_name=gesture_name).inc()
    
    def record_confidence(self, confidence: float):
        """Record prediction confidence score (MODEL METRIC)"""
        self.prediction_confidence.observe(confidence)
    
    def increment_data_quality(self, quality_type: str):
        """Record data quality indicators (DATA METRIC)"""
        self.input_data_quality.labels(quality_type=quality_type).inc()
    
    def increment_error(self, error_type: str):
        """Record API errors (SERVER METRIC)"""
        self.api_errors.labels(error_type=error_type).inc()
    
    def increment_maze_action(self, action: str):
        """Record maze game actions (SERVER METRIC)"""
        self.maze_actions.labels(action=action).inc()
    
    def get_timestamp(self) -> str:
        """Get current timestamp for health checks"""
        return datetime.utcnow().isoformat()
        self.gesture_predictions.labels(gesture_name=gesture_name).inc()
    
    def record_confidence(self, confidence: float):
        """Record prediction confidence score (MODEL METRIC)"""
        self.prediction_confidence.observe(confidence)
    
    def increment_data_quality(self, quality_type: str):
        """Record data quality indicators (DATA METRIC)"""
        self.input_data_quality.labels(quality_type=quality_type).inc()
    
    def increment_error(self, error_type: str):
        """Record API errors (SERVER METRIC)"""
        self.api_errors.labels(error_type=error_type).inc()
    
    def increment_maze_action(self, action: str):
        """Record maze game actions (SERVER METRIC)"""
        self.maze_actions.labels(action=action).inc()
    
    def get_timestamp(self) -> str:
        """Get current timestamp for health checks"""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
