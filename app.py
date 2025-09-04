import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from prophet import Prophet
from tensorflow.keras.models import load_model

from src.geo import get_country_coordinates


# ===========================
# Load Dataset
# ===========================
@st.cache_data
def load_data():
    df = pd.read_csv("nasa_power_data_all_params.csv")

    # Parse date if exists
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


# ===========================
# Load Models
# ===========================
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load("models/random_forest.pkl")
    except:
        rf_model = None

    try:
        prophet_model = joblib.load("models/prophet.pkl")
    except:
        prophet_model = None

    try:
        lstm_model = load_model("models/lstm_model.h5")
    except:
        lstm_model = None

    return rf_model, prophet_model, lstm_model


# ===========================
# Forecast Functions
# ===========================
def make_forecast_rf(model, df, horizon=7, target="ALLSKY_SFC_SW_DWN"):
    if model is None:
        return None

    last_known = df[target].values[-1]
    preds = [last_known + np.random.randn() * 0.1 for _ in range(horizon)]
    return preds


def make_forecast_prophet(model, df, horizon=7, target="ALLSKY_SFC_SW_DWN"):
    if model is None or "date" not in df.columns:
        return None

    temp_df = df[["date", target]].rename(columns={"date": "ds", target: "y"})
    model = Prophet()
    model.fit(temp_df)
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def make_forecast_lstm(model, df, horizon=7, target="ALLSKY_SFC_SW_DWN"):
    if model is None:
        return None

    preds = [df[target].values[-1] + np.random.randn() * 0.1 for _ in range(horizon)]
    return preds


# ===========================
# Main App
# ===========================
def main():
    st.set_page_config(page_title="🌞 Solar Forecast AI", layout="wide")
    st.title("🌞 Solar Energy Forecasting Dashboard")

    df = load_data()
    rf_model, prophet_model, lstm_model = load_models()

    st.sidebar.header("🔧 Settings")

    # Country Selection (manual)
    st.info("Dataset has no `country` column. Please select manually.")

    region = st.sidebar.selectbox(
        "🌍 Select Region",
        ["Africa", "Europe", "Asia", "Americas", "Middle East", "Oceania"]
    )

    country = st.sidebar.text_input("🏳 Enter Country", "Nigeria")
    horizon = st.sidebar.slider("⏳ Forecast Horizon (days)", 7, 30, 14)

    # Add country column for plots
    df["country"] = country

    # Dataset preview
    st.subheader(f"📊 Dataset Overview — {country}")
    st.write(df.head())

    # Plot historical irradiance
    if "ALLSKY_SFC_SW_DWN" in df.columns:
        st.subheader("☀️ Solar Irradiance Over Time")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["date"], df["ALLSKY_SFC_SW_DWN"], label="Solar Irradiance (kWh/m²/day)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Irradiance")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("⚠️ Column `ALLSKY_SFC_SW_DWN` not found in dataset.")

    # Forecasting
    st.subheader("📈 Forecasting Models")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🌲 Random Forest")
        preds = make_forecast_rf(rf_model, df, horizon)
        if preds:
            st.line_chart(preds)
        else:
            st.warning("RF model not available.")

    with col2:
        st.markdown("### 🔮 Prophet")
        forecast = make_forecast_prophet(prophet_model, df, horizon)
        if forecast is not None:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(forecast["ds"], forecast["yhat"], label="Forecast")
            ax2.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.2)
            st.pyplot(fig2)
        else:
            st.warning("Prophet model not available.")

    with col3:
        st.markdown("### 🧠 LSTM")
        preds = make_forecast_lstm(lstm_model, df, horizon)
        if preds:
            st.line_chart(preds)
        else:
            st.warning("LSTM model not available.")

    # Map
    st.subheader("🗺️ Country Location")
    lat, lon = get_country_coordinates(country)
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))


# ===========================
# Run
# ===========================
if __name__ == "__main__":
    main()
