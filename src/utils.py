# src/utils.py

import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
from src.constants import DATA_PATH, DEFAULT_DATE_COL, DEFAULT_TARGET_COL, DEFAULT_COUNTRY_COL, DATE_FORMAT


# ==========================
# Load Dataset
# ==========================
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load NASA POWER dataset and preprocess."""
    df = pd.read_csv(path)

    # Ensure date column is datetime
    if DEFAULT_DATE_COL in df.columns:
        df[DEFAULT_DATE_COL] = pd.to_datetime(df[DEFAULT_DATE_COL], errors="coerce")
        df = df.dropna(subset=[DEFAULT_DATE_COL])
    else:
        raise KeyError(f"{DEFAULT_DATE_COL} not found in dataset")

    # Add country column if missing (for region grouping in app)
    if DEFAULT_COUNTRY_COL not in df.columns:
        df[DEFAULT_COUNTRY_COL] = "Nigeria"  # default fallback

    # Sort by date
    df = df.sort_values(by=DEFAULT_DATE_COL).reset_index(drop=True)

    return df


# ==========================
# Load Models
# ==========================
def load_model_file(model_path: str, model_type: str = "pickle"):
    """Load a model (RandomForest, Prophet, or LSTM)."""
    if not os.path.exists(model_path):
        return None

    if model_type == "pickle":
        return joblib.load(model_path)
    elif model_type == "keras":
        return load_model(model_path)
    else:
        raise ValueError("Unsupported model type")


# ==========================
# Preprocess Data
# ==========================
def preprocess_data(df: pd.DataFrame, country: str = None) -> pd.DataFrame:
    """Filter by country and ensure target variable exists."""
    if country and DEFAULT_COUNTRY_COL in df.columns:
        df = df[df[DEFAULT_COUNTRY_COL] == country]

    if DEFAULT_TARGET_COL not in df.columns:
        raise KeyError(f"Target column {DEFAULT_TARGET_COL} not found in dataset")

    return df


# ==========================
# Forecasting Stub
# ==========================
def make_forecast(model, df: pd.DataFrame, horizon: int = 7):
    """
    Dummy forecast generator.
    Later can plug in RF/Prophet/LSTM models.
    """
    last_date = df[DEFAULT_DATE_COL].max()

    # Create forecast dates
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

    # Dummy forecast: repeat last known value
    last_val = df[DEFAULT_TARGET_COL].iloc[-1]
    preds = np.full(shape=horizon, fill_value=last_val)

    forecast_df = pd.DataFrame({
        DEFAULT_DATE_COL: future_dates,
        "Forecast": preds
    })

    return forecast_df
