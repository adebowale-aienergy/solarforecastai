import streamlit as st
import pandas as pd
from src.constants import (
    DATA_PATH, RF_MODEL_PATH, PROPHET_MODEL_PATH, LSTM_MODEL_PATH,
    DEFAULT_HORIZON, MIN_HORIZON, MAX_HORIZON
)
from src.utils import load_data, load_model, preprocess_data, make_forecast
from src.geo import get_country_regions, get_country_coordinates
from src.eval_utils import plot_evaluation_metrics

# ---------------------------
# App Title
# ---------------------------
st.title("🌍 Solar Energy Forecasting Dashboard")
st.write("Forecast solar energy generation using ML models (RF, Prophet, LSTM).")

# ---------------------------
# Sidebar Controls
# ---------------------------
regions = get_country_regions()
region = st.sidebar.selectbox("🌎 Select Region", list(regions.keys()))
country = st.sidebar.selectbox("🏳 Select Country", regions[region])

horizon = st.sidebar.slider("⏳ Forecast Horizon (days)", MIN_HORIZON, MAX_HORIZON, DEFAULT_HORIZON)
model_choice = st.sidebar.radio("🤖 Select Model", ["Random Forest", "Prophet", "LSTM"])

# ---------------------------
# Load Data
# ---------------------------
df = load_data()
df_country = preprocess_data(df, country, horizon)

# ---------------------------
# Load Model
# ---------------------------
if model_choice == "Random Forest":
    model = load_model(RF_MODEL_PATH, "sklearn")
    preds = make_forecast(model, "sklearn", df_country, horizon)
elif model_choice == "Prophet":
    model = load_model(PROPHET_MODEL_PATH, "prophet")
    preds = make_forecast(model, "prophet", df_country, horizon)
else:
    model = load_model(LSTM_MODEL_PATH, "lstm")
    preds = make_forecast(model, "lstm", df_country, horizon)

# ---------------------------
# Display Forecast
# ---------------------------
st.subheader(f"📊 Forecast for {country} ({model_choice})")
dates = df_country["DATE"].tail(horizon).values if "DATE" in df_country else range(horizon)

forecast_df = pd.DataFrame({"Date": dates, "Prediction": preds})
st.dataframe(forecast_df)

# ---------------------------
# Plot Results
# ---------------------------
if "TARGET" in df_country:
    y_true = df_country["TARGET"].values[:len(preds)]
    y_pred = preds[:len(y_true)]
    plot_evaluation_metrics(y_true, y_pred)
