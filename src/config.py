"""
Configuration settings for the loan approval model
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
# Try both locations for data file (root and data directory)
if (BASE_DIR / "loan_data.csv").exists():
    DATA_FILE = BASE_DIR / "loan_data.csv"
else:
    DATA_FILE = DATA_DIR / "loan_data.csv"
MODEL_FILE = MODEL_DIR / "model.pkl"
SCALER_FILE = MODEL_DIR / "scaler.pkl"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.25
MAX_AGE = 60
OUTLIER_THRESHOLD = 4
CORRELATION_THRESHOLD = 0.8

# Feature columns (after preprocessing)
FEATURE_COLUMNS = [
    'person_income',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'credit_score',
    'person_gender_male',
    'person_education_Bachelor',
    'person_education_Doctorate',
    'person_education_High School',
    'person_education_Master',
    'person_home_ownership_OTHER',
    'person_home_ownership_OWN',
    'person_home_ownership_RENT',
    'loan_intent_EDUCATION',
    'loan_intent_HOMEIMPROVEMENT',
    'loan_intent_MEDICAL',
    'loan_intent_PERSONAL',
    'loan_intent_VENTURE',
    'previous_loan_defaults_on_file_Yes'
]

# Categorical columns for encoding
CATEGORICAL_COLUMNS = [
    'person_gender',
    'person_education',
    'person_home_ownership',
    'loan_intent',
    'previous_loan_defaults_on_file'
]

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

