"""
Model training script for loan approval classification
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

from .config import DATA_FILE, MODEL_FILE, RANDOM_STATE, TEST_SIZE
from .preprocessing import (
    load_data, preprocess_data, prepare_features,
    transform_features, fit_scaler
)


def train_model():
    """
    Train the XGBoost model for loan approval prediction
    """
    print("Loading data...")
    df = load_data(DATA_FILE)
    print(f"Data loaded: {df.shape}")
    
    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    print(f"Data after preprocessing: {df_processed.shape}")
    
    # Prepare features and target
    X = prepare_features(df_processed)
    y = df_processed['loan_status']
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale features
    print("Scaling features...")
    X_train_scaled, scaler = transform_features(X_train, fit=True)
    X_test_scaled, _ = transform_features(X_test, scaler=scaler)
    
    # Train model
    print("Training XGBoost model...")
    model = XGBClassifier(random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("="*50)
    
    # Save model
    print(f"\nSaving model to {MODEL_FILE}...")
    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print("Model saved successfully!")
    
    return model, scaler, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


if __name__ == "__main__":
    train_model()

