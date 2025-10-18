# src/model_rf.py

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/random_forest_model.pkl")

def load_rf_model():
    """Load the Random Forest model."""
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        return model
    else:
        raise FileNotFoundError(f"Random Forest model not found at {MODEL_PATH}")

def predict_random_forest(features: pd.DataFrame):
    """
    Predict solar generation using the Random Forest model.
    Args:
        features (pd.DataFrame): Processed input features for prediction.
    Returns:
        np.ndarray: Predicted solar power output.
    """
    model = load_rf_model()
    predictions = model.predict(features)
    return predictions
