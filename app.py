import streamlit as st
import pandas as pd
import numpy as np
import sys, os

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.constants import (
    DATA_PATH, RF_MODEL_PATH, PROPHET_MODEL_PATH, LSTM_MODEL_PATH,
    DEFAULT_DATE_COL, DEFAULT_TARGET_COL, DEFAULT_COUNTRY_COL,
    DEFAULT_HORIZON, MIN_HORIZON, MAX_HORIZON
)
from src.utils import load_data, load_model, preprocess_data, make_forecast
from src.geo import get_country_regions, get_country_coordinates
from src.eval_utils import plot_evaluation_metrics


# ----------------------------
# MAIN APP
# ----------------------------
def main():
    st.title("🌞 Solar Energy Forecasting Dashboard")
    st.markdown("Forecast solar energy generation using NASA POWER dataset.")

    # Load dataset
    df = load_data(DATA_PATH)

    # Since dataset has no Country column, add a default one
    if "Country" not in df.columns:
        df["Country"] = "Nigeria"  # Adjust if dataset is for another region

    # Sidebar selections
    st.sidebar.header("🔧 Settings")

    # Country selection
    countries = df[DEFAULT_COUNTRY_COL].unique().tolist()
    country = st.sidebar.selectbox("🌍 Select Country", countries)

    # Region handling (dummy since only 1 country now)
    regions = get_country_regions([country])
    region = st.sidebar.selectbox("📍 Select Region", regions)

    # Forecast horizon
    horizon = st.sidebar.slider(
        "⏳ Forecast Horizon (days)", MIN_HORIZON, MAX_HORIZON, DEFAULT_HORIZON
    )

    # Model selection
    model_choice = st.sidebar.radio(
        "🤖 Choose Model", ["Random Forest", "Prophet", "LSTM"]
    )

    # Filter data
    country_data = df[df[DEFAULT_COUNTRY_COL] == country]

    st.subheader(f"📊 Historical Data - {country}")
    st.dataframe(country_data.head())

    # Preprocess
    X, y, scaler = preprocess_data(country_data)

    # Load selected model
    if model_choice == "Random Forest":
        model = load_model(RF_MODEL_PATH)
    elif model_choice == "Prophet":
        model = load_model(PROPHET_MODEL_PATH)
    else:
        model = load_model(LSTM_MODEL_PATH)

    # Forecast
    forecast = make_forecast(model, country_data, horizon, model_choice, scaler)

    st.subheader("🔮 Forecast Results")
    st.line_chart(forecast)

    # Evaluation metrics
    st.subheader("📈 Evaluation Metrics")
    plot_evaluation_metrics(y[-len(forecast):], forecast)

    # Map (dummy coordinates)
    lat, lon = get_country_coordinates(country)
    st.subheader("🗺️ Location")
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))


if __name__ == "__main__":
    main()
