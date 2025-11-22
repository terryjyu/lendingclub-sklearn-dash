import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import pathlib
import os

# Paths
CURRENT_DIR = pathlib.Path(__file__).parent
DATA_PATH = CURRENT_DIR.parent.parent.joinpath("datasets").resolve()
MODEL_PATH = CURRENT_DIR.joinpath("models").resolve()

# Ensure model directory exists
os.makedirs(MODEL_PATH, exist_ok=True)

def train():
    print("Loading data...")
    # Load data
    df = pd.read_csv(DATA_PATH.joinpath("lc_cleaned_combined.csv"), low_memory=False)
    
    # Features to use
    features = ['loan_amnt', 'term', 'grade', 'emp_length', 'home_ownership', 'annual_inc', 'purpose']
    target = 'loan_status' 
    
    # Preprocessing
    print("Preprocessing...")
    X = df[features].copy()
    y = df[target].copy()

    # Handle missing values
    X['annual_inc'] = X['annual_inc'].fillna(X['annual_inc'].mean())
    X['loan_amnt'] = X['loan_amnt'].fillna(X['loan_amnt'].mean())
    X['emp_length'] = X['emp_length'].fillna('Unknown')
    
    # Encode Categorical Variables
    encoders = {}
    cat_cols = ['term', 'grade', 'emp_length', 'home_ownership', 'purpose']
    
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    
    # Encode Target
    y_le = LabelEncoder()
    y = y_le.fit_transform(y.astype(str))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train LightGBM
    print("Training LightGBM...")
    model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    
    # Save
    print("Saving models...")
    joblib.dump(model, MODEL_PATH.joinpath("lightgbm_model.joblib"))
    joblib.dump(encoders, MODEL_PATH.joinpath("label_encoders.joblib"))
    joblib.dump(y_le, MODEL_PATH.joinpath("target_encoder.joblib"))
    print("Done!")

if __name__ == "__main__":
    train()
