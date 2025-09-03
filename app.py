import sys
import os

# ===========================
# Ensure src/ is in Python path
# ===========================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

# ===========================
# Imports
# ===========================
import streamlit as st
import pandas as pd
import numpy as np

from src.constants import (
    DATA_PATH, RF_MODEL_PATH, PROPHET_MODEL_PATH, LSTM_MODEL_PATH,
    DEFAULT_DATE_COL, DEFAULT_TARGET_COL, DEFAULT_COUNTRY_COL,
    DEFAULT_HORIZON, MIN_HORIZON, MAX_HORIZON
)
from src.utils import load_data, load_model, preprocess_data, make_forecast
from src.geo import get_country_regions, get_country_coordinates
from src.eval_utils import plot_evaluation_metrics

# ===========================
# App Title & Description
# ===========================
st.set_page_config(page_title="Solar Energy Forecasting", layout="wide")
st.title("🌍 Solar Energy Forecasting Dashboard")
st.write("Forecast solar energy generation using ML models (Random Forest, Prophet, LSTM).")

# ===========================
# Sidebar Controls
# ===========================
st.sidebar.header("⚙️ Configuration")

regions = get_country_regions()
region = st.sidebar.selectbox("🌎 Select Region", list(regions.keys()))
country = st.sidebar.selectbox("🏳 Select Country", regions[region])

horizon = st.sidebar.slider("⏳ Forecast Horizon (days)", MIN_HORIZON, MAX_HORIZON, DEFAULT_HORIZON)
model_choice = st.sidebar.radio("🤖 Select Model", ["Random Forest", "Prophet", "LSTM"])

# ===========================
# Load Data
# ===========================
df = load_data()

if country not in df[DEFAULT_COUNTRY_COL].unique():
    st.error(f"❌ No data available for {country}. Please choose another country.")
    st.stop()

df_country = preprocess_data(df, country, horizon)

# ===========================
# Load Model & Make Predictions
# ===========================
if model_choice == "Random Forest":
    model = load_model(RF_MODEL_PATH, "sklearn")
    preds = make_forecast(model, "sklearn", df_country, horizon)

elif model_choice == "Prophet":
    model = load_model(PROPHET_MODEL_PATH, "prophet")
    preds = make_forecast(model, "prophet", df_country, horizon)

else:  # LSTM
    model = load_mode_
