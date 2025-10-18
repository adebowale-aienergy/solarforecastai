# Save the app.py code to a file
app_code = """
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from huggingface_hub import hf_hub_download

# Print statements for debugging import paths
print("Attempting to import from src...")

# Correct the import paths to be relative to the app.py file
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
"""

with open("app.py", "w") as f:
    f.write(app_code)
