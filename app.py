import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from src.geo import get_country_regions, get_country_coordinates

# -----------------------------
# Paths to data and models
# -----------------------------
DATA_PATH = "nasa_power_data_all_params.csv"
RF_MODEL_PATH = "models/random_forest.pkl"
PROPHET_MODEL_PATH = "models/prophet_model.pkl"
LSTM_MODEL_PATH = "models/lstm_model.h5"

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    return df

# -----------------------------
# Load trained models
# -----------------------------
def load_model(path, model_type="pkl"):
    if not os.path.exists(path):
        return None
    if model_type == "pkl":
        return joblib.load(path)
    elif model_type == "h5":
        from tensorflow.keras.models import load_model
        return load_model(path)
    return None

# -----------------------------
# Forecast stub (can extend)
# -----------------------------
def make_forecast(model, X):
    try:
        return model.predict(X)
    except Exception:
        return np.zeros(len(X))

# -----------------------------
# Main App
