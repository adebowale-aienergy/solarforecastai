import sys
import os

# Ensure src/ is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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


# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def get_dataset():
    return pd.read_csv(DATA_PATH)


# ---------------------------
# App Layout
# ---------------------------
def main():
    st.set_page_config(page_title="Solar Energy Forecasting", layout="wide")
    st.title("☀️ Solar Energy Forecasting Dashboard")

    # Load dataset
    df = get_dataset()

    # Sidebar: Country & Model selection
    st.sidebar.header("Settings")

    regions = get_country_regions(df[DEFAULT_COUNTRY_COL].unique())
    selected_region = st.sidebar.selectbox("🌍 Select Region", list(regions.keys()))
    selected_country = st.sidebar.selectbox(
        "🏳️ Select Country", regions[selected_region]
    )

    model_choice = st.sidebar.radio(
        "🤖 Choose Forecast Model",
        ["Random Forest", "Prophet", "LSTM", "Baseline (None)"]
    )

    horizon = st.sidebar.slider(
        "⏳ Forecast Horizon (days)",
        min_value=MIN_HORIZON,
        max_value=MAX_HORIZON,
        value=DEFAULT_HORIZON
    )

    # Preprocess
    country_df = df[df[DEFAULT_COUNTRY_COL] == selected_country]
    if country_df.empty:
        st.warning(f"No data available for {selected_country}")
        return

    X, y = preprocess_data(country_df)

    # Load model
    if model_choice == "Random Forest":
        model = load_model(RF_MODEL_PATH)
    elif model_choice == "Prophet":
        model = load_model(PROPHET_MODEL_PATH)
    elif model_choice == "LSTM":
        model = load_model(LSTM_MODEL_PATH)
    else:
        model = None  # ✅ FIXED colon issue here

    # Forecast
    forecast_df = make_forecast(model, country_df, horizon)

    # Show results
    st.subheader(f"Forecast for {selected_country}")
    st.line_chart(forecast_df.set_index(DEFAULT_DATE_COL)[DEFAULT_TARGET_COL])

    # Evaluation
    st.subheader("📊 Model Evaluation")
    eval_fig = plot_evaluation_metrics(y, forecast_df[DEFAULT_TARGET_COL])
    st.pyplot(eval_fig)

    # Map
    st.subheader("🌍 Location Map")
    lat, lon = get_country_coordinates(selected_country)
    st.map(pd.DataFrame({"lat": [lat], "
