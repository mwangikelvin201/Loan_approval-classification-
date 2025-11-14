"""
Flask web application for loan approval prediction
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.predict import LoanPredictor
from src.config import MODEL_FILE
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize predictor
try:
    predictor = LoanPredictor()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    predictor = None


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict loan approval endpoint
    
    Expected JSON format:
    {
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
    """
    if predictor is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = [
            'person_age', 'person_gender', 'person_education',
            'person_income', 'person_emp_exp', 'person_home_ownership',
            'loan_amnt', 'loan_intent', 'loan_int_rate',
            'loan_percent_income', 'cb_person_cred_hist_length',
            'credit_score', 'previous_loan_defaults_on_file'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Convert to DataFrame for preprocessing
        df = pd.DataFrame([data])
        
        # Preprocess the data (encode categorical variables)
        from src.preprocessing import encode_categorical_features
        df_encoded = encode_categorical_features(df)
        
        # Make prediction
        result = predictor.predict(df_encoded)
        
        return jsonify({
            'success': True,
            'prediction': result
        }), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    
    Expected JSON format:
    {
        "applications": [
            { ... application data ... },
            { ... application data ... }
        ]
    }
    """
    if predictor is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data or 'applications' not in data:
            return jsonify({'error': 'No applications data provided'}), 400
        
        applications = data['applications']
        
        if not isinstance(applications, list):
            return jsonify({'error': 'Applications must be a list'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(applications)
        
        # Preprocess
        from src.preprocessing import encode_categorical_features
        df_encoded = encode_categorical_features(df)
        
        # Make predictions
        results = predictor.predict_batch(df_encoded)
        
        return jsonify({
            'success': True,
            'predictions': results,
            'count': len(results)
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'error': f'Batch prediction failed: {str(e)}'
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

