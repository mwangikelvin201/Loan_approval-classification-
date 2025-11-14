"""
Simple test script for the loan approval API
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_prediction():
    """Test prediction endpoint"""
    print("Testing prediction endpoint...")
    
    test_data = {
        "person_age": 25,
        "person_gender": "male",
        "person_education": "Bachelor",
        "person_income": 50000,
        "person_emp_exp": 3,
        "person_home_ownership": "RENT",
        "loan_amnt": 10000,
        "loan_intent": "EDUCATION",
        "loan_int_rate": 10.5,
        "loan_percent_income": 0.2,
        "cb_person_cred_hist_length": 3,
        "credit_score": 650,
        "previous_loan_defaults_on_file": "No"
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

if __name__ == "__main__":
    print("="*50)
    print("LOAN APPROVAL API TEST")
    print("="*50)
    print()
    
    try:
        test_health()
        test_prediction()
        print("="*50)
        print("Tests completed!")
        print("="*50)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API.")
        print("Make sure the Flask app is running on http://localhost:5000")
    except Exception as e:
        print(f"Error: {e}")

