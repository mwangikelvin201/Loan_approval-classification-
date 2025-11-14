"""
Data preprocessing utilities for loan approval model
"""
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from .config import (
    MAX_AGE, OUTLIER_THRESHOLD, CORRELATION_THRESHOLD,
    CATEGORICAL_COLUMNS, FEATURE_COLUMNS, SCALER_FILE
)


def load_data(file_path):
    """Load the loan data from CSV file"""
    return pd.read_csv(file_path)


def clean_data(df):
    """
    Clean the dataset by removing outliers and invalid records
    """
    # Remove age outliers (cap at MAX_AGE)
    df = df[df['person_age'] <= MAX_AGE].copy()
    
    # Remove z-score outliers
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(zscore(df[numerical_cols]))
    outlier_mask = (z_scores > OUTLIER_THRESHOLD).any(axis=1)
    df_clean = df[~outlier_mask].copy()
    
    return df_clean


def encode_categorical_features(df):
    """
    Encode categorical variables using one-hot encoding
    """
    df_encoded = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
        dtype=int
    )
    return df_encoded


def remove_highly_correlated_features(df, target_col='loan_status'):
    """
    Remove highly correlated features
    """
    df_features = df.drop(columns=[target_col], errors='ignore')
    corr_matrix = df_features.corr()
    
    # Find highly correlated pairs
    corr_pairs = corr_matrix.unstack().sort_values(kind="quicksort", ascending=False)
    high_corr_pairs = [
        (i, j) for i, j in corr_pairs.index 
        if i != j and abs(corr_pairs[i, j]) > CORRELATION_THRESHOLD
    ]
    
    # Drop one variable from each pair
    columns_to_drop = set()
    for i, j in high_corr_pairs:
        if j in df.columns and j not in columns_to_drop:
            columns_to_drop.add(j)
    
    df_clean = df.drop(columns=columns_to_drop)
    return df_clean


def preprocess_data(df, fit_scaler=True, scaler=None):
    """
    Complete preprocessing pipeline
    """
    # Clean data
    df_clean = clean_data(df)
    
    # Encode categorical features
    df_encoded = encode_categorical_features(df_clean)
    
    # Remove highly correlated features
    df_final = remove_highly_correlated_features(df_encoded)
    
    # Ensure loan_status is integer
    if 'loan_status' in df_final.columns:
        df_final['loan_status'] = df_final['loan_status'].astype(int)
    
    return df_final


def prepare_features(df, feature_columns=None):
    """
    Prepare features for model training/prediction
    """
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS
    
    # Select only the features that exist in the dataframe
    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features].copy()
    
    # Add missing features with zeros
    missing_features = [col for col in feature_columns if col not in df.columns]
    for feature in missing_features:
        X[feature] = 0
    
    # Reorder columns to match expected order
    X = X[feature_columns]
    
    return X


def fit_scaler(X_train):
    """Fit and save the scaler"""
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Save scaler
    SCALER_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(scaler, SCALER_FILE)
    
    return scaler


def load_scaler():
    """Load the saved scaler"""
    if SCALER_FILE.exists():
        return joblib.load(SCALER_FILE)
    return None


def transform_features(X, scaler=None, fit=False):
    """
    Scale features using StandardScaler
    """
    if scaler is None:
        scaler = load_scaler()
    
    if scaler is None and fit:
        scaler = fit_scaler(X)
    
    if scaler is None:
        raise ValueError("Scaler not found. Please train the model first or provide a scaler.")
    
    X_scaled = scaler.transform(X)
    return X_scaled, scaler

