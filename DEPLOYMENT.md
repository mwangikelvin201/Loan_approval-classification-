# Deployment Guide

This guide explains how to deploy the Loan Approval Classification system.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Docker and Docker Compose (optional, for containerized deployment)

## Local Deployment

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Ensure your `loan_data.csv` file is in the `data/` directory:

```bash
mkdir -p data
# Copy your loan_data.csv to data/ directory
```

### 3. Train the Model

Before running the application, you need to train the model:

```bash
python train.py
```

This will:
- Load and preprocess the data
- Train the XGBoost model
- Save the model to `models/model.pkl`
- Save the scaler to `models/scaler.pkl`
- Display model performance metrics

### 4. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### 5. Access the Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

## Docker Deployment

### 1. Build and Run with Docker Compose

```bash
docker-compose up --build
```

This will:
- Build the Docker image
- Start the container
- Make the application available on port 5000

### 2. Build and Run with Docker

```bash
# Build the image
docker build -t loan-approval-app .

# Run the container
docker run -p 5000:5000 -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data loan-approval-app
```

## API Endpoints

### Health Check
```
GET /health
```

Returns the health status of the application.

### Single Prediction
```
POST /predict
Content-Type: application/json

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
```

### Batch Prediction
```
POST /predict/batch
Content-Type: application/json

{
    "applications": [
        { ... application 1 ... },
        { ... application 2 ... }
    ]
}
```

## Production Deployment

### Using Gunicorn

For production, use Gunicorn instead of Flask's development server:

```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 app:app
```

### Environment Variables

You can set the following environment variables:

- `FLASK_ENV`: Set to `production` for production mode
- `PORT`: Port number (default: 5000)

### Reverse Proxy (Nginx)

Example Nginx configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Cloud Deployment

### AWS EC2

1. Launch an EC2 instance
2. Install Docker or Python
3. Clone the repository
4. Follow the deployment steps above
5. Configure security groups to allow port 5000

### Heroku

1. Install Heroku CLI
2. Create a `Procfile`:
   ```
   web: gunicorn app:app
   ```
3. Deploy:
   ```bash
   heroku create
   git push heroku main
   ```

### Google Cloud Platform

1. Use Cloud Run for containerized deployment
2. Build and push Docker image to Container Registry
3. Deploy to Cloud Run

## Monitoring

- Health check endpoint: `/health`
- Check application logs for errors
- Monitor model performance metrics
- Set up alerts for model drift

## Troubleshooting

### Model Not Found Error

If you see "Model not loaded" error:
1. Ensure you've run `python train.py` first
2. Check that `models/model.pkl` and `models/scaler.pkl` exist
3. Verify file permissions

### Port Already in Use

If port 5000 is already in use:
- Change the port in `app.py` or use environment variable
- Or stop the process using port 5000

### Import Errors

If you encounter import errors:
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're using the correct Python version (3.8+)

## Security Considerations

- Use HTTPS in production
- Implement authentication/authorization
- Validate and sanitize all inputs
- Rate limit API endpoints
- Keep dependencies updated
- Use environment variables for sensitive data

