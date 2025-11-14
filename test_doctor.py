"""
Test prediction for a doctor (not married) with previous loan defaults
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.predict import LoanPredictor
import pandas as pd

print("="*70)
print("LOAN PREDICTION: DOCTOR (NOT MARRIED) WITH PREVIOUS LOAN DEFAULTS")
print("="*70)
print()

try:
    # Initialize predictor
    predictor = LoanPredictor()
    
    # Doctor loan application with previous defaults
    doctor_application = {
        'person_age': 35,  # Typical age after medical school and residency
        'person_gender': 'male',  # Can be changed to 'female'
        'person_education': 'Doctorate',  # Doctor has doctorate degree
        'person_income': 180000,  # High income typical for doctors
        'person_emp_exp': 8,  # Several years of practice
        'person_home_ownership': 'MORTGAGE',  # Doctor likely owns home
        'loan_amnt': 25000,  # Reasonable loan amount
        'loan_intent': 'MEDICAL',  # Medical-related loan purpose
        'loan_int_rate': 12.0,  # Higher rate due to previous defaults
        'loan_percent_income': 0.14,  # Low percentage of income
        'cb_person_cred_hist_length': 10,  # Long credit history
        'credit_score': 620,  # Lower score due to defaults
        'previous_loan_defaults_on_file': 'Yes'  # KEY RISK FACTOR
    }
    
    print("APPLICANT PROFILE:")
    print("-" * 70)
    print(f"  Age:                           {doctor_application['person_age']} years")
    print(f"  Gender:                        {doctor_application['person_gender']}")
    print(f"  Education:                     {doctor_application['person_education']} (Medical Doctor)")
    print(f"  Marital Status:                Not Married (Single)")
    print(f"  Annual Income:                 ${doctor_application['person_income']:,}")
    print(f"  Employment Experience:         {doctor_application['person_emp_exp']} years")
    print(f"  Home Ownership:                {doctor_application['person_home_ownership']}")
    print(f"  Credit Score:                  {doctor_application['credit_score']}")
    print(f"  Credit History Length:         {doctor_application['cb_person_cred_hist_length']} years")
    print(f"  Previous Defaults:             {doctor_application['previous_loan_defaults_on_file']} [WARNING]")
    print()
    print("LOAN DETAILS:")
    print("-" * 70)
    print(f"  Loan Amount:                   ${doctor_application['loan_amnt']:,}")
    print(f"  Loan Purpose:                  {doctor_application['loan_intent']}")
    print(f"  Interest Rate:                 {doctor_application['loan_int_rate']}%")
    print(f"  Loan as % of Income:           {doctor_application['loan_percent_income']*100:.0f}%")
    print()
    
    # Preprocess the data
    from src.preprocessing import encode_categorical_features
    df = pd.DataFrame([doctor_application])
    df_encoded = encode_categorical_features(df)
    
    # Make prediction
    print("Processing application...")
    print()
    result = predictor.predict(df_encoded)
    
    print("="*70)
    print("PREDICTION RESULT")
    print("="*70)
    print()
    print(f"  DECISION:                      {result['status']}")
    print(f"  Confidence Level:              {result['confidence']*100:.2f}%")
    print()
    print("  Probability Breakdown:")
    print(f"    - Approval Probability:      {result['probability']['approved']*100:.2f}%")
    print(f"    - Rejection Probability:     {result['probability']['rejected']*100:.2f}%")
    print()
    
    if result['status'] == 'Approved':
        print("  [APPROVED] This loan application has been APPROVED.")
        print("  Despite previous defaults, the applicant meets approval criteria.")
    else:
        print("  [REJECTED] This loan application has been REJECTED.")
        print("  Previous loan defaults are a significant risk factor.")
    
    print()
    print("="*70)
    print("RISK ANALYSIS")
    print("="*70)
    print()
    print("Positive Factors:")
    print("  + High income ($180,000)")
    print("  + Doctorate education")
    print("  + Long employment experience (8 years)")
    print("  + Long credit history (10 years)")
    print("  + Low loan-to-income ratio (14%)")
    print("  + Home ownership (Mortgage)")
    print()
    print("Negative Factors:")
    print("  [WARNING] PREVIOUS LOAN DEFAULTS - Major red flag")
    print("  [WARNING] Lower credit score (620) due to defaults")
    print("  [WARNING] Higher interest rate (12%)")
    print()
    print("Key Question: Do high income and education offset the risk")
    print("              of previous defaults?")
    print()
    
except FileNotFoundError as e:
    print(f"[ERROR] {e}")
    print("\nPlease train the model first by running: python train.py")
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

