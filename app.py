import streamlit as st
import pandas as pd
import numpy as np
import os

from src.data_utils import load_dataset
from src.model_utils import load_models, make_prediction
from src.visualization import plot_forecast, plot_actual_vs_predicted
from src.geo import add_country_column, get_country_regions

# ===========================
# Paths
# ===========================
DATA_PATH = "nasa_power_data_all_params.csv"
MODELS = {
    "Random Forest": "models/rf_model.pkl",
    "Prophet": "models/prophet_model.pkl",
    "LSTM": "models/lstm_model.h5",
}

# ===========================
# Main App
# ===========================
def main():
    st.set_page_config(
        page_title="SolarForecastAI",
        page_icon="☀️",
        layout="wide"
    )

    # Sidebar
    with st.sidebar:
        st.image("assets/logo.png", use_container_width=True)
        st.markdown("### 🌞 SolarForecastAI")
        st.markdown("Forecasting solar energy with ML & AI")

    # Load dataset
    df = load_dataset(DATA_PATH)

    # Ensure "country" column exists
    df = add_country_column(df)

    # Sidebar filters
    regions = get_country_regions(df["country"].unique())
    region = st.sidebar.selectbox("🌍 Select Region", list(regions.keys()))
    country = st.sidebar.selectbox("🏳️ Select Country", regions[region])
    model_choice = st.sidebar.radio("🤖 Select Model", list(MODELS.keys()))

    st.sidebar.markdown("---")
    horizon = st.sidebar.slider("🔮 Forecast Horizon (days)", 1, 30, 7)

    # Title
    st.title("☀️ Solar Forecast Dashboard")
    st.markdown(f"### Forecasting Solar Energy for **{country}** using {model_choice}")

    # Filter dataset by country
    country_df = df[df["country"] == country].copy()

    if country_df.empty:
        st.error(f"No data available for {country}. Try another country.")
        return

    # Load selected model
    models = load_models(MODELS)
    model = models[model_choice]

    # Make forecast
    forecast_df = make_prediction(model_choice, model, country_df, horizon=horizon)

    # Layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Forecast vs Actual")
        fig1 = plot_forecast(country_df, forecast_df)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("📊 Actual vs Predicted")
        fig2 = plot_actual_vs_predicted(country_df, forecast_df)
        st.plotly_chart(fig2, use_container_width=True)

    # Show raw forecast
    st.subheader("🔎 Forecast Data")
    st.dataframe(forecast_df.head(20))


# ===========================
# Run app
# ===========================
if __name__ == "__main__":
    main()
