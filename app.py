import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure src/ is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    load_data, preprocess_data,
    load_model, make_forecast,
    train_random_forest, train_prophet, train_lstm
)
from src.constants import (
    DATA_PATH, RF_MODEL_PATH, PROPHET_MODEL_PATH, LSTM_MODEL_PATH,
    DEFAULT_TARGET_COL, DEFAULT_HORIZON, MIN_HORIZON, MAX_HORIZON
)

# ===========================
# Load & Preprocess Data
# ===========================
@st.cache_data
def get_data():
    df = load_data(DATA_PATH)
    return preprocess_data(df)

df = get_data()

st.title("☀️ Solar Energy Forecasting Dashboard")
st.markdown("Forecast solar radiation using ML models (Random Forest, Prophet, LSTM).")

# ===========================
# Sidebar
# ===========================
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Choose Forecast Model",
    ["Baseline", "Random Forest", "Prophet", "LSTM"]
)

horizon = st.sidebar.slider(
    "Forecast Horizon (days)", MIN_HORIZON, MAX_HORIZON, DEFAULT_HORIZON
)

target = st.sidebar.selectbox(
    "Select Target Variable", [DEFAULT_TARGET_COL] + [c for c in df.columns if c not in ["Date"]]
)

# ===========================
# Train or Load Models
# ===========================
if model_choice == "Random Forest":
    if not os.path.exists(RF_MODEL_PATH):
        st.sidebar.info("Training Random Forest model...")
        model = train_random_forest(df, target=target, save_path=RF_MODEL_PATH)
    else:
        model = load_model(RF_MODEL_PATH)

elif model_choice == "Prophet":
    if not os.path.exists(PROPHET_MODEL_PATH):
        st.sidebar.info("Training Prophet model...")
        model = train_prophet(df, target=target, save_path=PROPHET_MODEL_PATH)
    else:
        model = load_model(PROPHET_MODEL_PATH)

elif model_choice == "LSTM":
    if not os.path.exists(LSTM_MODEL_PATH):
        st.sidebar.info("Training LSTM model (may take a while)...")
        model = train_lstm(df, target=target, save_path=LSTM_MODEL_PATH)
    else
