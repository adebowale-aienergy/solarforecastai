# src/model_prophet.py

from prophet import Prophet
import pandas as pd
from pathlib import Path
import joblib

MODEL_PATH = Path("models/prophet_model.pkl")

def load_prophet_model():
    """Load the saved Prophet model."""
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        return model
    else:
        raise FileNotFoundError(f"Prophet model not found at {MODEL_PATH}")

def predict_prophet(future_df: pd.DataFrame):
    """
    Predict solar energy using Prophet model.
    Args:
        future_df (pd.DataFrame): DataFrame with 'ds' column for datetime.
    Returns:
        pd.DataFrame: Forecast results with 'ds' and 'yhat'.
    """
    model = load_prophet_model()
    forecast = model.predict(future_df)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
