"""
Prediction utilities for loan approval model
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from .config import MODEL_FILE, FEATURE_COLUMNS
from .preprocessing import load_scaler, transform_features, prepare_features


class LoanPredictor:
    """Loan approval prediction class"""
    
    def __init__(self, model_path=None, scaler=None):
        """
        Initialize the predictor with model and scaler
        """
        if model_path is None:
            model_path = MODEL_FILE
        
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Please train the model first using train_model.py"
            )
        
        self.model = joblib.load(model_path)
        self.scaler = scaler if scaler else load_scaler()
        self.feature_columns = FEATURE_COLUMNS
        
        if self.scaler is None:
            raise ValueError(
                "Scaler not found. Please ensure the scaler was saved during training."
            )
    
    def predict(self, data):
        """
        Predict loan approval for given data
        
        Args:
            data: pandas DataFrame or dict with loan application data
            
        Returns:
            dict with prediction, probability, and status
        """
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Prepare features
        X = prepare_features(data, self.feature_columns)
        
        # Scale features
        X_scaled, _ = transform_features(X, self.scaler)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        return {
            'prediction': int(prediction),
            'probability': {
                'approved': float(probability[1]),
                'rejected': float(probability[0])
            },
            'status': 'Approved' if prediction == 1 else 'Rejected',
            'confidence': float(max(probability))
        }
    
    def predict_batch(self, data):
        """
        Predict loan approval for multiple applications
        
        Args:
            data: pandas DataFrame with multiple loan applications
            
        Returns:
            list of prediction dictionaries
        """
        # Prepare features
        X = prepare_features(data, self.feature_columns)
        
        # Scale features
        X_scaled, _ = transform_features(X, self.scaler)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        results = []
        for i in range(len(predictions)):
            results.append({
                'prediction': int(predictions[i]),
                'probability': {
                    'approved': float(probabilities[i][1]),
                    'rejected': float(probabilities[i][0])
                },
                'status': 'Approved' if predictions[i] == 1 else 'Rejected',
                'confidence': float(max(probabilities[i]))
            })
        
        return results

