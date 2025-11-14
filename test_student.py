"""
Test prediction for a 25-year-old male student applying for a luxury loan
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.predict import LoanPredictor
import pandas as pd

print("="*70)
print("LOAN PREDICTION: 25-YEAR-OLD MALE STUDENT - LUXURY LOAN")
print("="*70)
print()

try:
    # Initialize predictor
    predictor = LoanPredictor()
    
    # Student loan application for luxury purposes
    student_application = {
        'person_age': 25,
        'person_gender': 'male',
        'person_education': 'Bachelor',  # Assuming college student
        'person_income': 25000,  # Low income typical for students
        'person_emp_exp': 1,  # Limited work experience
        'person_home_ownership': 'RENT',  # Student likely renting
        'loan_amnt': 10000,  # Loan amount for luxury purchase
        'loan_intent': 'PERSONAL',  # Closest to luxury (no specific luxury category)
        'loan_int_rate': 14.5,  # Higher rate for student/riskier borrower
        'loan_percent_income': 0.4,  # 40% of income - high ratio
        'cb_person_cred_hist_length': 2,  # Short credit history
        'credit_score': 580,  # Lower credit score typical for students
        'previous_loan_defaults_on_file': 'No'  # Assuming no defaults
    }
    
    print("APPLICANT PROFILE:")
    print("-" * 70)
    print(f"  Age:                           {student_application['person_age']} years")
    print(f"  Gender:                        {student_application['person_gender']}")
    print(f"  Education:                     {student_application['person_education']} (Student)")
    print(f"  Annual Income:                 ${student_application['person_income']:,}")
    print(f"  Employment Experience:         {student_application['person_emp_exp']} year(s)")
    print(f"  Home Ownership:                {student_application['person_home_ownership']}")
    print(f"  Credit Score:                  {student_application['credit_score']}")
    print(f"  Credit History Length:         {student_application['cb_person_cred_hist_length']} years")
    print(f"  Previous Defaults:             {student_application['previous_loan_defaults_on_file']}")
    print()
    print("LOAN DETAILS:")
    print("-" * 70)
    print(f"  Loan Amount:                   ${student_application['loan_amnt']:,}")
    print(f"  Loan Purpose:                  {student_application['loan_intent']} (Luxury)")
    print(f"  Interest Rate:                 {student_application['loan_int_rate']}%")
    print(f"  Loan as % of Income:           {student_application['loan_percent_income']*100:.0f}%")
    print()
    
    # Preprocess the data
    from src.preprocessing import encode_categorical_features
    df = pd.DataFrame([student_application])
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
        print("  The applicant meets the criteria for loan approval.")
    else:
        print("  [REJECTED] This loan application has been REJECTED.")
        print("  The applicant does not meet the risk criteria for approval.")
    
    print()
    print("="*70)
    print("RISK ANALYSIS")
    print("="*70)
    print()
    print("Key Risk Factors Identified:")
    print("  - Low income relative to loan amount")
    print("  - High loan-to-income ratio (40%)")
    print("  - Limited employment experience")
    print("  - Lower credit score")
    print("  - Short credit history")
    print("  - Loan purpose: Personal/Luxury (higher risk category)")
    print()
    
except FileNotFoundError as e:
    print(f"[ERROR] {e}")
    print("\nPlease train the model first by running: python train.py")
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

