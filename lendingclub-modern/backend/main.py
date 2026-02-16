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
    data_path = os.path.join(os.path.dirname(__file__), "../../datasets/lc_cleaned_combined.csv")
    
    # Load Models
    try:
        models["lgbm"] = joblib.load(os.path.join(model_path, "lightgbm_model.joblib"))
        models["encoders"] = joblib.load(os.path.join(model_path, "label_encoders.joblib"))
        print("LightGBM model and encoders loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

    # Load and Preprocess Data for Stats
    try:
        print("Loading pre-calculated stats...")
        # Load lightweight JSONs instead of 700MB CSV
        models["stats_map"] = pd.read_json(os.path.join(model_path, "stats_map.json"), orient="records")
        
        # Load sample data for boxplots
        try:
             models["sample"] = pd.read_json(os.path.join(model_path, "sample.json"), orient="records")
        except:
             models["sample"] = []

        models["data"] = None 
        
        print("Stats loaded successfully (Optimized mode).")
        
    except Exception as e:
        print(f"Error loading stats: {e}")

@app.get("/")
def read_root():
    return {"message": "LendingClub Prediction API is running"}

@app.get("/stats/region")
def get_region_stats():
    if "stats_region" not in models:
        raise HTTPException(status_code=503, detail="Data not loaded")
    df = models["stats_region"]
    return df.to_dict(orient="records")

@app.get("/stats/state")
def get_state_stats():
    if "stats_state" not in models:
        raise HTTPException(status_code=503, detail="Data not loaded")
    df = models["stats_state"]
    return df.to_dict(orient="records")

@app.get("/stats/map")
def get_map_stats():
    if "stats_map" not in models:
        raise HTTPException(status_code=503, detail="Data not loaded")
    df = models["stats_map"]
    return df.to_dict(orient="records")

@app.get("/stats/sample")
def get_sample_data(limit: int = 1000):
    """Get a random sample of raw data for boxplots to avoid sending 100MB"""
    if "sample" not in models:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Return pre-calculated sample
    df = models["sample"]
    if isinstance(df, list): return df # Handle empty case
    
    return df.head(limit).to_dict(orient="records")

@app.post("/predict")
def predict_loan(application: LoanApplication):
    if not models or "lgbm" not in models:
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
                try:
                    user_df[col] = le.transform(user_df[col].astype(str))
                except ValueError as e:
                     raise HTTPException(status_code=400, detail=f"Invalid value for {col}: {e}")
            else:
                 pass

        # Ensure column order matches training
        expected_cols = ['loan_amnt', 'term', 'grade', 'emp_length', 'home_ownership', 'annual_inc', 'purpose']
        user_df = user_df[expected_cols]

        if application.loan_amnt > 40000:
            # Hard rule: LendingClub does not issue loans > $40k
            prob = 0.0
            result = {
                "probability": 0.0,
                "approval_chance": "0.00%",
                "message": f"Rejected: The requested amount of ${application.loan_amnt:,.0f} exceeds LendingClub's maximum limit of $40,000."
            }
            return result

        # Predict using LightGBM
        lgbm_model = models["lgbm"]
        prob = lgbm_model.predict_proba(user_df)[0][1]

        # Formulate response
        result = {
            "probability": float(prob),
            "approval_chance": f"{prob:.2%}",
            "message": f"You have {prob:.2%} chance of getting a loan amount of ${application.loan_amnt:,.0f}."
        }

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
