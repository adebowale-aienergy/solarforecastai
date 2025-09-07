import streamlit as st
import pandas as pd
import joblib
import numpy as np
from src.data_utils import load_dataset
from src.model_utils import load_models, make_prediction
from src.visualization import plot_forecast, plot_actual_vs_predicted
from src.geo import get_country_regions, get_country_coordinates

# ----------------------------
# Sidebar Branding
# ----------------------------
with st.sidebar:
    st.image("assets/logo.png", use_container_width=True)
    st.markdown("### 🌞 SolarForecastAI")
    st.markdown("Forecasting solar energy with ML & AI")

# ----------------------------
# Main App
# ----------------------------
def main():
    st.title("☀️ Solar Energy Forecasting Dashboard")

    # Load dataset
    df = load_dataset("nasa_power_data_all_params.csv")

    # Regions & Countries
    countries = df["country"].unique()
    regions = get_country_regions(countries)

    region = st.sidebar.selectbox("🌍 Select Region", list(regions.keys()))
    country = st.sidebar.selectbox("🏳️ Select Country", regions[region])
    lat, lon = get_country_coordinates(country)

    st.sidebar.markdown(f"**Coordinates:** {lat}, {lon}")

    # Load Models
    models = load_models("models")

    # Model Selection
    model_choice = st.sidebar.radio("📊 Select Model", list(models.keys()))

    # Forecast Horizon
    forecast_days = st.sidebar.slider("⏳ Forecast Horizon (days)", 1, 30, 7)

    # Subset Data
    country_data = df[df["country"] == country]

    # Predictions
    y_true, y_pred, forecast_df = make_prediction(
        models[model_choice], country_data, forecast_days
    )

    # ----------------------------
    # Layout: Two Columns
    # ----------------------------
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Forecast Visualization")
        st.plotly_chart(plot_forecast(forecast_df, country), use_container_width=True)
        st.subheader("🔍 Actual vs Predicted")
        st.plotly_chart(plot_actual_vs_predicted(y_true, y_pred, country), use_container_width=True)

    with col2:
        st.subheader("📋 Key Forecast Info")
        st.metric("Forecast Days", forecast_days)
        st.metric("Selected Model", model_choice)
        st.metric("Region", region)
        st.metric("Country", country)

        st.markdown("### Forecast Data Preview")
        st.dataframe(forecast_df.head())

if __name__ == "__main__":
    main()
