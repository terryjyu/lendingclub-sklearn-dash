import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import pathlib
import os

# Define paths
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("datasets").resolve()
MODEL_PATH = PATH.joinpath("models").resolve()

print(f"Data path: {DATA_PATH}")
print(f"Model path: {MODEL_PATH}")

# Create models directory if it doesn't exist
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

print("Loading data...")
try:
    df = pd.read_csv(DATA_PATH.joinpath("lc_cleaned_combined.csv"), low_memory=False)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

print("Preparing data...")
# Select columns as per original script
x = df[['loan_amnt', 'term', 'grade', 'emp_length', 'home_ownership', 'annual_inc', 'purpose']]
y = df.loan_status

# Split data
print("Splitting data...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0)

# Label Encoding
print("Encoding data and saving encoders...")
encoders = {}
for col in ['term', 'grade', 'emp_length', 'home_ownership', 'purpose']:
    le = LabelEncoder()
    # Fit on train
    le.fit(x_train[col].astype(str))
    # Transform train
    x_train[col] = le.transform(x_train[col].astype(str))
    
    # Save the encoder
    encoders[col] = le
    
    # Transform test (handling potential unseen labels by ignoring or simple try/except if needed)
    try:
        x_test[col] = le.transform(x_test[col].astype(str))
    except Exception as e:
        print(f"Warning: Unseen labels in {col} for test set. Skipping test transform for this column.")

# Save the dictionary of encoders
joblib.dump(encoders, MODEL_PATH.joinpath("label_encoders.joblib"))
print("Label encoders saved.")

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=50, oob_score=True, random_state=123456)
rf.fit(x_train, y_train)

print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)

print("Saving models...")
joblib.dump(rf, MODEL_PATH.joinpath("sklearn_rf.joblib"), compress=3)
joblib.dump(lr, MODEL_PATH.joinpath("sklearn_lr.joblib"), compress=3)

print("Models and encoders saved successfully.")
