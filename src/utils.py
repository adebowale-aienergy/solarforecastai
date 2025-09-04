# src/utils.py

import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from src.constants import (
    DATA_PATH, DEFAULT_DATE_COL, DEFAULT_TARGET_COL, DEFAULT_COUNTRY_COL
)

def load_data():
    """Load dataset and standardize column names to uppercase."""
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.upper()  # force uppercase
    return df

def load_model(model_path, model_type="sklearn"):
    """Load models depending on type."""
    if model_type == "sklearn" or model_type == "prophet":
        return joblib.load(model_path)
    elif model_type == "lstm":
        return load_model(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def preprocess_data(df):
    """Preprocess dataset (ensure uppercase, drop NA)."""
    df.columns = df.columns.str.upper()
    df = df.dropna()
    
    # Ensure required columns exist
    for col in [DEFAULT_DATE_COL, DEFAULT_TARGET_COL, DEFAULT_COUNTRY_COL]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
    
    return df

def make_forecast(model, df, horizon=7, model_type="sklearn"):
    """Generate forecast from model."""
    if model_type == "sklearn":
        X = df.drop(columns=[DEFAULT_TARGET_COL, DEFAULT_DATE_COL, DEFAULT_COUNTRY_COL])
        preds = model.predict(X[-horizon:])
        return preds

    elif model_type == "prophet":
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        return forecast[['ds', 'yhat']].tail(horizon)

    elif model_type == "lstm":
        X = df.drop(columns=[DEFAULT_TARGET_COL, DEFAULT_DATE_COL, DEFAULT_COUNTRY_COL])
        X = np.array(X[-horizon:]).reshape((1, horizon, X.shape[1]))
        preds = model.predict(X)
        return preds.flatten()

    else:
        raise ValueError(f"Unknown model type: {model_type}")
