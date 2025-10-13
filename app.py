# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from huggingface_hub import hf_hub_download

from src.fetch_data import (
    load_clean_data, load_features_data, load_merged_data
)
from src.preprocess import ensure_datetime, filter_country, get_rf_features, prepare_lstm_sequences
from src.visualization import plot_time_series_streamlit, plot_forecast_map_streamlit
from src.evaluation import metrics


# ---- PAGE SETTINGS ----
st.set_page_config(layout="wide", page_title="Solar Forecasting Dashboard")


# ---- LOAD MODELS FROM HUGGING FACE (ONCE, CACHED) ----
@st.cache_resource(show_spinner=True)
def load_models():
    try:
        rf_path = hf_hub_download(
            repo_id="adebowale-aienergy/solarforecastai",
            filename="models/random_forest_model.pkl"
        )
        prophet_path = hf_hub_download(
            repo_id="adebowale-aienergy/solarforecastai",
            filename="models/prophet_model.pkl"
        )
        lstm_path = hf_hub_download(
            repo_id="adebowale-aienergy/solarforecastai",
            filename="models/lstm_model.h5"
        )

        rf_model = joblib.load(rf_path)
        prophet_model = joblib.load(prophet_path)
        lstm_model = tf.keras.models.load_model(lstm_path)

        return rf_model, prophet_model, lstm_model

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None


rf_model, prophet_model, lstm_model = load_models()


# ---- LOAD DATASETS (CACHED) ----
@st.cache_data(show_spinner=False)
def cached_load_clean():
    return load_clean_data()

@st.cache_data(show_spinner=False)
def cached_load_features():
    return load_features_data()

@st.cache_data(show_spinner=False)
def cached_load_merged():
    return load_merged_data()


# ---- SIDEBAR ----
st.sidebar.title("Solar ForecastAI 🌞")
region = st.sidebar.selectbox("Region", ["Africa", "Europe", "Asia", "Americas", "Middle East", "Oceania"])
country = st.sidebar.selectbox("Country", [
    "Nigeria", "Kenya", "South Africa", "Egypt", "Morocco",
    "Germany", "France", "United Kingdom", "Spain", "Italy",
    "China", "India", "Japan", "South Korea", "Saudi Arabia",
    "United States", "Brazil", "Mexico", "Canada", "Argentina",
    "UAE", "Israel", "Qatar", "Iran", "Turkey", "Australia",
    "New Zealand", "Fiji", "Papua New Guinea", "Samoa"
])
model_choice = st.sidebar.selectbox("Model", ["Random Forest", "Prophet", "LSTM"])
download_btn = st.sidebar.button("Cache Data and Models")


if download_btn:
    st.info("Caching datasets and models, please wait...")
    _ = cached_load_clean()
    _ = cached_load_features()
    _ = cached_load_merged()
    _ = load_models()
    st.success("All datasets and models cached successfully ✅")


# ---- MAIN TITLE ----
st.title("☀️ Solar Forecasting Dashboard")


# ---- LOAD MAIN DATA ----
clean_df = cached_load_clean()
features_df = cached_load_features()

if clean_df is None or clean_df.empty:
    st.warning("⚠️ Clean data could not be loaded.")
else:
    df_country = filter_country(clean_df, "country", country)
    df_country = ensure_datetime(df_country, "date")

    # ---- SUMMARY METRICS ----
    st.markdown("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg GHI", f"{df_country['GHI'].mean():.2f}")
    col2.metric("Avg Temp (°C)", f"{df_country['temp'].mean():.1f}")
    col3.metric("Predicted PV (kWh/day)", f"{df_country['solar_irradiance'].mean():.2f}")
    col4.metric("CO₂ Avoided (kg)", "—")

    st.markdown("---")

    # ---- FORECAST VISUALS ----
    left, right = st.columns([2, 3])

    with left:
        st.markdown("#### Forecast Map")
        agg = df_country.groupby("country").agg({"solar_irradiance": "mean"}).reset_index().rename(columns={"solar_irradiance": "forecast"})
        if not agg.empty:
            plot_forecast_map_streamlit(agg, "forecast")
        else:
            st.info("No forecast data for selected country.")

    with right:
        st.markdown("#### Forecast Time Series")

        if model_choice == "Random Forest" and rf_model:
            try:
                X = get_rf_features(df_country)
                preds = rf_model.predict(X)
                df_ts = df_country.copy()
                df_ts["forecast"] = preds
                plot_time_series_streamlit(df_ts, "date", "forecast", "Random Forest Forecast")
            except Exception as e:
                st.error(f"Random Forest error: {e}")

        elif model_choice == "Prophet" and prophet_model:
            try:
                dfp = df_country.rename(columns={"date": "ds", "solar_irradiance": "y"})[["ds"]].copy()
                forecast = prophet_model.predict(dfp)
                df_ts = df_country.copy()
                df_ts["forecast"] = forecast["yhat"].values[:len(df_ts)]
                plot_time_series_streamlit(df_ts, "date", "forecast", "Prophet Forecast")
            except Exception as e:
                st.error(f"Prophet error: {e}")

        elif model_choice == "LSTM" and lstm_model:
            try:
                series = df_country.sort_values("date")["solar_irradiance"].dropna()
                lookback = 24
                X_seq, y_seq, scales = prepare_lstm_sequences(series, lookback=lookback)
                preds_scaled = lstm_model.predict(X_seq).flatten()
                mn, mx = scales["min"], scales["max"]
                preds = preds_scaled * (mx - mn) + mn
                df_ts = df_country.iloc[lookback:].copy()
                df_ts["forecast"] = preds[:len(df_ts)]
                plot_time_series_streamlit(df_ts, "date", "forecast", "LSTM Forecast")
            except Exception as e:
                st.error(f"LSTM error: {e}")
        else:
            st.info("Model not available or not selected.")

    st.markdown("---")

    # ---- MODEL COMPARISON ----
    st.markdown("### Quick Model Comparison")
    cols = st.columns(3)

    with cols[0]:
        st.subheader("Random Forest")
        if rf_model is not None:
            st.success("Ready")
        else:
            st.warning("Not loaded")

    with cols[1]:
        st.subheader("Prophet")
        if prophet_model is not None:
            st.success("Ready")
        else:
            st.warning("Not loaded")

    with cols[2]:
        st.subheader("LSTM")
        if lstm_model is not None:
            st.success("Ready")
        else:
            st.warning("Not loaded")

st.caption("🔗 Data and Models Hosted on Hugging Face | Built by Adebowale AI Energy")
