"""Streamlit app tying everything together."""
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
    DEFAULT_HORIZON, MIN_HORIZON, MAX_HORIZON,
)
from src.data_utils import load_data, filter_by_country, add_time_features, make_prophet_frame, split_features_target
from src.geo_utils import countries_by_region, get_country_coords
from src.model_utils import (
    load_rf_model, load_prophet_model, load_lstm_model, make_forecast
)
from src.eval_utils import regression_metrics
from src.visualization import (
    preview_table, line_actual_vs_pred, prophet_forecast_plot, model_comparison_plot, country_map
)
from src.utils import get_countries_by_region

st.set_page_config(page_title="SolarForecastAI", layout="wide")
st.title("☀️ SolarForecastAI")
st.markdown("Forecast solar generation using NASA POWER data and ML models (Random Forest, Prophet, LSTM).")

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.header("Controls")
region = st.sidebar.selectbox("🌍 Select Region", list(countries_by_region.keys()))
countries = get_countries_by_region(region)
country = st.sidebar.selectbox("🏳️ Select Country", countries)
model_choice = st.sidebar.radio("🔀 Select Model", ["Random Forest", "Prophet", "LSTM"])
horizon = st.sidebar.slider("⏳ Forecast Horizon (days)", MIN_HORIZON, MAX_HORIZON, DEFAULT_HORIZON)

# -----------------------
# Load data
# -----------------------
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Data file not found at '{DATA_PATH}'. Place your CSV there or update src/constants.py.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.subheader("📊 Dataset preview")
st.dataframe(preview_table(df, n=8))

# Map visualization (use coordinates from constants)
lat, lon = get_country_coords(country)
st.subheader(f"🗺 Location: {country}")
st.plotly_chart(country_map(lat, lon, country), use_container_width=True)

# Filter by country if available in the dataset
df_country = filter_by_country(df, country, country_col=DEFAULT_COUNTRY_COL)

# Check target & date columns
if DEFAULT_TARGET_COL not in df_country.columns:
    st.warning(f"Target column '{DEFAULT_TARGET_COL}' not found in dataset. App expects that column. Update src/constants or your CSV.")
# Show available columns for debugging
st.caption(f"Columns: {', '.join(df_country.columns)}")

# -----------------------
# Load model
# -----------------------
MODEL_PATHS = {
    "Random Forest": RF_MODEL_PATH,
    "Prophet": PROPHET_MODEL_PATH,
    "LSTM": LSTM_MODEL_PATH,
}

model_path = MODEL_PATHS.get(model_choice)
model = None
try:
    if model_choice == "Random Forest":
        model = load_rf_model(model_path)
    elif model_choice == "Prophet":
        model = load_prophet_model(model_path)
    else:
        model = load_lstm_model(model_path)
except FileNotFoundError:
    st.error(f"Model file not found at '{model_path}'. Make sure your models are saved in the models/ folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model '{model_choice}': {e}")
    st.stop()

# -----------------------
# Forecast / Predict
# -----------------------
st.subheader("📈 Forecast / Prediction")
try:
    out = make_forecast(
        model=model,
        model_name=model_choice,
        df=df_country,
        horizon=horizon,
        date_col=DEFAULT_DATE_COL,
        target_col=DEFAULT_TARGET_COL,
    )
except Exception as e:
    st.error(f"Error during forecasting: {e}")
    st.stop()

# Display results
if out.get("forecast_df") is not None:
    # Prophet path
    st.info("Prophet forecast displayed below.")
    forecast_df = out["forecast_df"]
    # Optional: pass historical ds/y if available in df_country
    history_df = None
    if DEFAULT_DATE_COL in df_country.columns and DEFAULT_TARGET_COL in df_country.columns:
        history_df = make_prophet_frame(df_country, date_col=DEFAULT_DATE_COL, target_col=DEFAULT_TARGET_COL)
    fig = prophet_forecast_plot(forecast_df, history_df=history_df)
    st.plotly_chart(fig, use_container_width=True)
else:
    y_true = out.get("y_true")
    y_pred = out.get("y_pred")
    if y_true is None or y_pred is None:
        st.warning("Model did not return predictions (check data, model compatibility, or target column).")
    else:
        st.write("### Actual vs Predicted (holdout/test set)")
        fig = line_actual_vs_pred(y_true, y_pred, title=f"{model_choice} - Actual vs Predicted")
        st.plotly_chart(fig, use_container_width=True)

        # Metrics
        metrics = regression_metrics(y_true, y_pred)
        st.write("### Metrics")
        st.metric("MAE", metrics["MAE"])
        st.metric("RMSE", metrics["RMSE"])
        st.metric("MAPE (%)", metrics["MAPE"])

# -----------------------
# Comparison (attempt to run all models)
# -----------------------
st.subheader("🔍 Model Comparison")
comparison_series = {}
for mname, mpath in MODEL_PATHS.items():
    try:
        # load model (skip if same as selected — already loaded, but it's fine)
        if mname == "Random Forest":
            m = load_rf_model(mpath)
            outm = make_forecast(m, "Random Forest", df_country, horizon=horizon, date_col=DEFAULT_DATE_COL, target_col=DEFAULT_TARGET_COL)
        elif mname == "Prophet":
            m = load_prophet_model(mpath)
            outm = make_forecast(m, "Prophet", df_country, horizon=horizon, date_col=DEFAULT_DATE_COL, target_col=DEFAULT_TARGET_COL)
        else:
            m = load_lstm_model(mpath)
            outm = make_forecast(m, "LSTM", df_country, horizon=horizon, date_col=DEFAULT_DATE_COL, target_col=DEFAULT_TARGET_COL)
    except Exception:
        continue

    if outm.get("y_pred") is not None:
        # use first N points of predictions (they may differ in length)
        comparison_series[mname] = outm["y_pred"]

if comparison_series:
    figc = model_comparison_plot(comparison_series, title="Model Predictions Comparison")
    st.plotly_chart(figc, use_container_width=True)
else:
    st.info("No model predictions available for comparison. Check that models exist and are compatible with dataset features.")

# Footer
st.markdown("---")
st.markdown("Project by Adebowale Immanuel Adeyemi — SolarForecastAI")
