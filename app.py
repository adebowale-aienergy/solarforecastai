import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# PATHS
# =========================
DATA_PATH = "nasa_power_data_all_params.csv"
RF_MODEL_PATH = "models/random_forest.pkl"
PROPHET_MODEL_PATH = "models/prophet_model.pkl"
LSTM_MODEL_PATH = "models/lstm_model.h5"

# =========================
# REGION → COUNTRY MAPPING
# =========================
REGION_COUNTRY_MAP = {
    "Africa": ["Nigeria", "Kenya", "South Africa", "Ghana"],
    "Europe": ["Germany", "France", "Norway", "UK"],
    "Asia": ["China", "India", "Japan"],
    "Americas": ["USA", "Brazil", "Canada"],
    "Middle East": ["UAE", "Saudi Arabia", "Qatar"],
    "Oceania": ["Australia", "New Zealand"]
}

# =========================
# LOADERS
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_resource
def load_models():
    rf = pickle.load(open(RF_MODEL_PATH, "rb"))
    prophet = pickle.load(open(PROPHET_MODEL_PATH, "rb"))
    lstm = load_model(LSTM_MODEL_PATH)
    return rf, prophet, lstm

# =========================
# FORECAST FUNCTION
# =========================
def make_forecast(model, model_name, df, horizon=30):
    df = df.copy()
    if model_name == "Random Forest":
        X = np.arange(len(df), len(df) + horizon).reshape(-1, 1)
        preds = model.predict(X)
        future_dates = pd.date_range(df["date"].max(), periods=horizon + 1, freq="D")[1:]
        return pd.DataFrame({"date": future_dates, "forecast": preds})

    elif model_name == "Prophet":
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        return forecast[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "forecast"})

    elif model_name == "LSTM":
        data = df["target"].values.reshape(-1, 1)
        seq_len = 10
        x_input = data[-seq_len:].reshape(1, seq_len, 1)
        preds = []
        for _ in range
