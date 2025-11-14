# Quick Start Guide

Get the Loan Approval Classification System up and running in minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Train the Model

Before you can make predictions, you need to train the model:

```bash
python train.py
```

This will:
- Load and preprocess the data from `data/loan_data.csv`
- Train an XGBoost model
- Save the model to `models/model.pkl`
- Save the scaler to `models/scaler.pkl`
- Display model performance metrics

**Expected output:**
- Training takes a few minutes
- You'll see accuracy, precision, recall, and F1-score metrics
- Model files will be saved in the `models/` directory

## Step 3: Run the Web Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## Step 4: Use the System

### Option A: Web Interface
1. Open your browser
2. Navigate to `http://localhost:5000`
3. Fill in the loan application form
4. Click "Predict Loan Approval"
5. View the prediction result

### Option B: Python API
```python
from src.predict import LoanPredictor

predictor = LoanPredictor()
result = predictor.predict({
    'person_age': 25,
    'person_gender': 'male',
    # ... other fields
})
print(result)
```

### Option C: REST API
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 25,
    "person_gender": "male",
    ...
  }'
```

## Testing

Test the API with:
```bash
python test_api.py
```

## Docker Quick Start

If you prefer Docker:

```bash
# Build and run
docker-compose up --build
```

The application will be available at `http://localhost:5000`

## Troubleshooting

### "Model not found" error
- Make sure you've run `python train.py` first
- Check that `models/model.pkl` exists

### Port 5000 already in use
- Change the port in `app.py` or stop the other service

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Use Python 3.8 or higher

## Next Steps

- Read [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment
- Check [README.md](README.md) for full documentation
- See `example_usage.py` for code examples

