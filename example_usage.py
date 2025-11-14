"""
Example usage of the LoanPredictor class
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.predict import LoanPredictor
import pandas as pd

def example_single_prediction():
    """Example of making a single prediction"""
    print("="*60)
    print("Example: Single Loan Prediction")
    print("="*60)
    
    # Initialize predictor
    predictor = LoanPredictor()
    
    # Example loan application data
    application = {
        'person_age': 25,
        'person_gender': 'male',
        'person_education': 'Bachelor',
        'person_income': 50000,
        'person_emp_exp': 3,
        'person_home_ownership': 'RENT',
        'loan_amnt': 10000,
        'loan_intent': 'EDUCATION',
        'loan_int_rate': 10.5,
        'loan_percent_income': 0.2,
        'cb_person_cred_hist_length': 3,
        'credit_score': 650,
        'previous_loan_defaults_on_file': 'No'
    }
    
    # Make prediction
    result = predictor.predict(application)
    
    print(f"\nApplication Details:")
    for key, value in application.items():
        print(f"  {key}: {value}")
    
    print(f"\nPrediction Result:")
    print(f"  Status: {result['status']}")
    print(f"  Confidence: {result['confidence']*100:.2f}%")
    print(f"  Approval Probability: {result['probability']['approved']*100:.2f}%")
    print(f"  Rejection Probability: {result['probability']['rejected']*100:.2f}%")
    print()

def example_batch_prediction():
    """Example of making batch predictions"""
    print("="*60)
    print("Example: Batch Loan Predictions")
    print("="*60)
    
    # Initialize predictor
    predictor = LoanPredictor()
    
    # Example batch of loan applications
    applications = [
        {
            'person_age': 30,
            'person_gender': 'female',
            'person_education': 'Master',
            'person_income': 75000,
            'person_emp_exp': 5,
            'person_home_ownership': 'MORTGAGE',
            'loan_amnt': 15000,
            'loan_intent': 'HOMEIMPROVEMENT',
            'loan_int_rate': 8.5,
            'loan_percent_income': 0.2,
            'cb_person_cred_hist_length': 5,
            'credit_score': 720,
            'previous_loan_defaults_on_file': 'No'
        },
        {
            'person_age': 22,
            'person_gender': 'male',
            'person_education': 'High School',
            'person_income': 25000,
            'person_emp_exp': 1,
            'person_home_ownership': 'RENT',
            'loan_amnt': 5000,
            'loan_intent': 'PERSONAL',
            'loan_int_rate': 15.0,
            'loan_percent_income': 0.2,
            'cb_person_cred_hist_length': 2,
            'credit_score': 550,
            'previous_loan_defaults_on_file': 'Yes'
        }
    ]
    
    # Convert to DataFrame and preprocess
    df = pd.DataFrame(applications)
    from src.preprocessing import encode_categorical_features
    df_encoded = encode_categorical_features(df)
    
    # Make predictions
    results = predictor.predict_batch(df_encoded)
    
    print(f"\nProcessing {len(applications)} applications...\n")
    
    for i, (app, result) in enumerate(zip(applications, results), 1):
        print(f"Application {i}:")
        print(f"  Applicant: {app['person_age']} year old {app['person_gender']}")
        print(f"  Income: ${app['person_income']:,.0f}")
        print(f"  Loan Amount: ${app['loan_amnt']:,.0f}")
        print(f"  Credit Score: {app['credit_score']}")
        print(f"  Result: {result['status']} (Confidence: {result['confidence']*100:.2f}%)")
        print()

if __name__ == "__main__":
    try:
        example_single_prediction()
        print("\n")
        example_batch_prediction()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease train the model first by running: python train.py")
    except Exception as e:
        print(f"Error: {e}")

