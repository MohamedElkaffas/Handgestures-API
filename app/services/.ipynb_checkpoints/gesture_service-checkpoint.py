"""
Gesture prediction service with proper 2D preprocessing
Matches the training pipeline preprocessing
FIXED preprocessing to match training
"""

import joblib
import numpy as np
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GestureService:
    """Service for hand gesture prediction with CORRECT preprocessing"""
    
    def __init__(self, model_path: str, encoder_path: str):
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.model = None
        self.label_encoder = None
        
        # Gesture to maze action mapping
        self.gesture_to_action = {
            'like': 'UP',           
            'dislike': 'DOWN',      
            'one': 'LEFT',          
            'rock': 'RIGHT',         
            'palm': 'STOP',         
            'fist': 'WAIT',         
            'peace': 'PAUSE',       
            'ok': 'OK',             
            'call': 'ACTION',       
            'stop': 'STOP',         
            'mute': 'MUTE',         
            'four': 'FOUR',         
            'three': 'THREE',       
            'two_up': 'TWO',        
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load the trained model and label encoder using joblib"""
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded with joblib from {self.model_path}")
            
            self.label_encoder = joblib.load(self.encoder_path)
            logger.info(f"Label encoder loaded with joblib from {self.encoder_path}")
            
            # Check model input requirements
            if hasattr(self.model, 'n_features_in_'):
                logger.info(f"Model expects {self.model.n_features_in_} features")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.model = None
            self.label_encoder = None
    
    def preprocess_landmarks(self, landmarks: List[float]) -> np.ndarray:
        """
        FIXED: Preprocess landmarks to match training pipeline
        Input: 63 values (21 landmarks × 3 coordinates)
        Output: 42 values (21 landmarks × 2 coordinates) - ONLY x,y
        """
        if len(landmarks) != 63:
            raise ValueError(f"Expected 63 landmarks, got {len(landmarks)}")
        
        # Convert to numpy array and reshape to (21, 3)
        landmarks_array = np.array(landmarks, dtype=np.float32).reshape(21, 3)
        
        # CRITICAL FIX: Extract only x,y coordinates (drop z)
        # Model was trained with 42 features = 21 landmarks × 2 coordinates
        xy_coordinates = landmarks_array[:, :2]  # Only x,y, drop z
        
        # Flatten to 42 features for model input
        processed = xy_coordinates.flatten()
        
        logger.info(f"Preprocessed {len(landmarks)} -> {len(processed)} features")
        return processed
    
    def predict(self, landmarks: List[float]) -> Dict[str, Any]:
        """Predict gesture with CORRECT preprocessing"""
        if not self.model or not self.label_encoder:
            raise RuntimeError("Model not loaded properly")
        
        try:
            # Apply CORRECTED preprocessing (63 -> 42 features)
            processed_landmarks = self.preprocess_landmarks(landmarks)
            
            # Reshape for model prediction (1, 42)
            input_data = processed_landmarks.reshape(1, -1)
            
            logger.info(f"Input shape: {input_data.shape}")
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            prediction_proba = self.model.predict_proba(input_data)[0]
            
            # Get gesture name
            gesture_name = self.label_encoder.inverse_transform([prediction])[0]
            
            # Get confidence
            confidence = float(np.max(prediction_proba))
            
            # Get maze action
            maze_action = self.gesture_to_action.get(gesture_name, 'WAIT')
            
            return {
                'gesture_name': gesture_name,
                'maze_action': maze_action,
                'confidence': confidence,
                'prediction_number': int(prediction)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def health_check(self) -> bool:
        """Check if the service is healthy"""
        return self.model is not None and self.label_encoder is not None