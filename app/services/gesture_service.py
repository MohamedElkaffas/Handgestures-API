"""
Gesture prediction service with proper 2D preprocessing
Matches the training pipeline preprocessing
"""

import joblib  
import pickle
import numpy as np
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GestureService:
    """Service for hand gesture prediction with proper joblib loading"""
    
    def __init__(self, model_path: str, encoder_path: str):
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.model = None
        self.label_encoder = None
        
        # Gesture to maze action mapping
        self.gesture_to_action = {
            'thumbs_up': 'UP',
            'thumbs_down': 'DOWN', 
            'pointing_left': 'LEFT',
            'pointing_right': 'RIGHT',
            'open_hand': 'STOP',
            'fist': 'WAIT',
            'peace': 'PAUSE',
            'ok_sign': 'OK'
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load the trained model and label encoder using joblib"""
        try:
            # Load the trained model with joblib (not pickle!)
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded with joblib from {self.model_path}")
            
            # Load the label encoder with joblib  
            self.label_encoder = joblib.load(self.encoder_path)
            logger.info(f"Label encoder loaded with joblib from {self.encoder_path}")
            
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            self.model = None
            self.label_encoder = None
        except Exception as e:
            logger.error(f"Error loading models with joblib: {e}")
            self.model = None
            self.label_encoder = None
    
    def preprocess_landmarks(self, landmarks: List[float]) -> np.ndarray:
        """
        Preprocess landmarks to match training pipeline
        Input: 63 values (21 landmarks × 3 coordinates)
        Output: 2D array (21, 3) -> flattened for model
        """
        if len(landmarks) != 63:
            raise ValueError(f"Expected 63 landmarks, got {len(landmarks)}")
        
        # Convert to numpy array
        landmarks_array = np.array(landmarks, dtype=np.float32)
        
        # Reshape to 2D format (21 landmarks, 3 coordinates each)
        landmarks_2d = landmarks_array.reshape(21, 3)
        
        # Validate normalized coordinates (should be 0-1 for x,y)
        if np.any(landmarks_2d[:, :2] < 0) or np.any(landmarks_2d[:, :2] > 1):
            logger.warning("Some landmarks outside [0,1] range")
        
        # Flatten back for model input (most models expect 1D)
        return landmarks_2d.flatten()
    
    def predict(self, landmarks: List[float]) -> Dict[str, Any]:
        """
        Predict gesture from landmarks with proper preprocessing
        """
        if not self.model or not self.label_encoder:
            raise RuntimeError("Model not loaded properly")
        
        try:
            # Apply the same preprocessing as training
            processed_landmarks = self.preprocess_landmarks(landmarks)
            
            # Reshape for model prediction (1, 63)
            input_data = processed_landmarks.reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            prediction_proba = self.model.predict_proba(input_data)[0]
            
            # Get gesture name
            gesture_name = self.label_encoder.inverse_transform([prediction])[0]
            
            # Get confidence (max probability)
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