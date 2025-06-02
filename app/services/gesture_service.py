"""
Gesture recognition business logic service
"""

import joblib
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json

from app.utils.preprocessing import process_hand_landmarks_xy

class GestureService:
    """Service for gesture recognition and maze control logic"""
    
    def __init__(self, model_path: str, encoder_path: str):
        self.model_path = Path(model_path)
        self.encoder_path = Path(encoder_path)
        self.model = None
        self.label_encoder = None
        self.gesture_mappings = None
        self._load_resources()
    
    def _load_resources(self):
        """Load model, encoder, and gesture mappings"""
        try:
            # Load model
            self.model = joblib.load(self.model_path)
            
            # Load label encoder
            with open(self.encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Load gesture mappings
            mappings_path = Path("app/data/gesture_mappings.json")
            with open(mappings_path) as f:
                self.gesture_mappings = json.load(f)
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model resources: {e}")
    
    def predict(self, landmarks: list) -> dict:
        """
        Predict gesture from landmarks
        """
        try:
            # Preprocess landmarks
            features = process_hand_landmarks_xy(pd.Series(landmarks)).reshape(1, -1)
            
            # Get predictions
            pred_numeric = self.model.predict(features)[0]
            pred_proba = self.model.predict_proba(features)[0]
            confidence = float(max(pred_proba))
            
            # Convert to gesture name
            gesture_name = self.label_encoder.inverse_transform([pred_numeric])[0]
            
            # Get maze action
            maze_action = self.gesture_mappings["maze_controls"].get(gesture_name, "UNKNOWN")
            
            return {
                "gesture_name": gesture_name,
                "maze_action": maze_action,
                "confidence": confidence,
                "prediction_number": int(pred_numeric)
            }
            
        except Exception as e:
            raise ValueError(f"Prediction failed: {e}")
    
    def health_check(self) -> bool:
        """Check if model is loaded and functional"""
        try:
            return all([
                self.model is not None,
                self.label_encoder is not None,
                self.gesture_mappings is not None
            ])
        except:
            return False
