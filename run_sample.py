"""
Simple sample script to test the loan prediction system
"""
import sys
import os
from pathlib import Path

print("="*60)
print("LOAN APPROVAL PREDICTION SYSTEM - SAMPLE RUN")
print("="*60)
print()

# Check if model exists
model_path = Path("models/model.pkl")
if not model_path.exists():
    print("[X] Model not found. Training model first...")
    print("   Run: python train.py")
    sys.exit(1)
else:
    print("[OK] Model file found:", model_path)
    print()

# Check if scaler exists
scaler_path = Path("models/scaler.pkl")
if not scaler_path.exists():
    print("[!] Scaler not found. Model may not work correctly.")
    print("   Run: python train.py")
    print()
else:
    print("[OK] Scaler file found:", scaler_path)
    print()

# Try to import and run prediction
try:
    print("Loading prediction module...")
    from src.predict import LoanPredictor
    
    print("Initializing predictor...")
    predictor = LoanPredictor()
    print("[OK] Predictor initialized successfully!")
    print()
    
    # Sample loan application
    print("="*60)
    print("SAMPLE PREDICTION")
    print("="*60)
    print()
    
    sample_application = {
        'person_age': 28,
        'person_gender': 'female',
        'person_education': 'Bachelor',
        'person_income': 65000,
        'person_emp_exp': 4,
        'person_home_ownership': 'RENT',
        'loan_amnt': 12000,
        'loan_intent': 'EDUCATION',
        'loan_int_rate': 9.5,
        'loan_percent_income': 0.18,
        'cb_person_cred_hist_length': 4,
        'credit_score': 680,
        'previous_loan_defaults_on_file': 'No'
    }
    
    print("Application Details:")
    print("-" * 60)
    for key, value in sample_application.items():
        print(f"  {key:30s}: {value}")
    print()
    
    print("Making prediction...")
    result = predictor.predict(sample_application)
    
    print()
    print("="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print()
    print(f"Status:           {result['status']}")
    print(f"Confidence:       {result['confidence']*100:.2f}%")
    print(f"Approval Prob:    {result['probability']['approved']*100:.2f}%")
    print(f"Rejection Prob:   {result['probability']['rejected']*100:.2f}%")
    print()
    
    if result['status'] == 'Approved':
        print("[APPROVED] LOAN APPROVED!")
    else:
        print("[REJECTED] LOAN REJECTED")
    print()
    
    # Test with a second sample (high risk)
    print("="*60)
    print("SAMPLE PREDICTION #2 (High Risk Applicant)")
    print("="*60)
    print()
    
    high_risk_application = {
        'person_age': 22,
        'person_gender': 'male',
        'person_education': 'High School',
        'person_income': 25000,
        'person_emp_exp': 1,
        'person_home_ownership': 'RENT',
        'loan_amnt': 15000,
        'loan_intent': 'PERSONAL',
        'loan_int_rate': 18.0,
        'loan_percent_income': 0.6,
        'cb_person_cred_hist_length': 2,
        'credit_score': 520,
        'previous_loan_defaults_on_file': 'Yes'
    }
    
    print("Application Details:")
    print("-" * 60)
    for key, value in high_risk_application.items():
        print(f"  {key:30s}: {value}")
    print()
    
    print("Making prediction...")
    result2 = predictor.predict(high_risk_application)
    
    print()
    print("="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print()
    print(f"Status:           {result2['status']}")
    print(f"Confidence:       {result2['confidence']*100:.2f}%")
    print(f"Approval Prob:    {result2['probability']['approved']*100:.2f}%")
    print(f"Rejection Prob:   {result2['probability']['rejected']*100:.2f}%")
    print()
    
    if result2['status'] == 'Approved':
        print("[APPROVED] LOAN APPROVED!")
    else:
        print("[REJECTED] LOAN REJECTED")
    print()
    
    print("="*60)
    print("[SUCCESS] SAMPLE RUN COMPLETED SUCCESSFULLY!")
    print("="*60)
    
except ImportError as e:
    print(f"[ERROR] Import Error: {e}")
    print()
    print("Please install dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

