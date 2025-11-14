#!/usr/bin/env python
"""
Standalone script to train the loan approval model
Run this script to train and save the model before deploying the application
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.train_model import train_model

if __name__ == "__main__":
    print("="*60)
    print("LOAN APPROVAL MODEL TRAINING")
    print("="*60)
    print()
    
    try:
        model, scaler, metrics = train_model()
        print()
        print("="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nModel and scaler have been saved.")
        print("You can now run the Flask application using: python app.py")
        print("Or use Docker: docker-compose up")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)

