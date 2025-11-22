from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from typing import Optional

app = FastAPI(title="LendingClub Prediction API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all. In production, specify frontend URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
models = {}

class LoanApplication(BaseModel):
    term: str
    loan_amnt: float
    grade: str
    emp_length: str
    home_ownership: str
    annual_inc: float
    purpose: str

@app.on_event("startup")
def load_models():
    model_path = os.path.join(os.path.dirname(__file__), "models")
    try:
        models["lgbm"] = joblib.load(os.path.join(model_path, "lightgbm_model.joblib"))
        models["encoders"] = joblib.load(os.path.join(model_path, "label_encoders.joblib"))
        print("LightGBM model and encoders loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        # In production, you might want to crash here if models are critical

@app.get("/")
def read_root():
    return {"message": "LendingClub Prediction API is running"}

@app.post("/predict")
def predict_loan(application: LoanApplication):
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        # Create DataFrame
        data = {
            'loan_amnt': [application.loan_amnt],
            'term': [application.term],
            'grade': [application.grade],
            'emp_length': [application.emp_length],
            'home_ownership': [application.home_ownership],
            'annual_inc': [application.annual_inc],
            'purpose': [application.purpose]
        }
        user_df = pd.DataFrame(data)

        # Encode features
        encoders = models["encoders"]
        for col in ['term', 'grade', 'home_ownership', 'purpose', 'emp_length']:
            if col in encoders:
                le = encoders[col]
                # Handle unseen labels strictly or gracefully? 
                # For now, we assume frontend sends valid options.
                # But we need to cast to string as encoders expect strings
                try:
                    user_df[col] = le.transform(user_df[col].astype(str))
                except ValueError as e:
                     raise HTTPException(status_code=400, detail=f"Invalid value for {col}: {e}")
            else:
                 # Should not happen if encoders are correct
                 pass

        # Ensure column order matches training
        # Based on page2.py: ['loan_amnt', 'term', 'grade', 'emp_length', 'home_ownership', 'annual_inc', 'purpose']
        expected_cols = ['loan_amnt', 'term', 'grade', 'emp_length', 'home_ownership', 'annual_inc', 'purpose']
        user_df = user_df[expected_cols]

        # Predict using LightGBM
        lgbm_model = models["lgbm"]
        prob = lgbm_model.predict_proba(user_df)[0][1]

        # Formulate response
        result = {
            "probability": float(prob),
            "approval_chance": f"{prob:.2%}",
            "message": ""
        }

        if application.loan_amnt < 1000 or application.loan_amnt > 40000:
             result["message"] = f"Although Lending Club only offers loans between $1000 and $40000, according to our ML prediction, you might have {result['approval_chance']} chance."
        else:
             result["message"] = f"You have {result['approval_chance']} chance of getting a loan amount of ${application.loan_amnt}."

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
