import streamlit as st
import pandas as pd
from src.data_utils import load_data, filter_by_country, add_time_features, make_prophet_frame
from src.geo_utils import countries_by_region, get_country_coords
from src.model_utils import (
    load_rf_model,
    load_prophet_model,
    load_lstm_model,
    make_forecast,
)
from src.eval_utils import regression_metrics
from src.visualization import preview_table, line_actual_vs_pred, prophet_forecast_plot, model_comparison_plot, country_map
from src.constants import DEFAULT_DATE_COL, DEFAULT_TARGET_COL

st.set_page_config(page_title="SolarForecastAI", layout="wide")
st.title("☀️ SolarForecastAI")
st.markdown("Forecast solar generation using NASA POWER data and machine learning models (Random Forest, Prophet, LSTM).")

# Sidebar
st.sidebar.header("Controls")
region = st.sidebar.selectbox("🌍 Region", list(countries_by_region.__self__.REGION_COUNTRIES.keys()) if False else list(countries_by_region.__code__.co_consts[1]) if False else ["Africa","Europe","Asia","Americas","Middle East","Oceania"])
# The above line attempts to be robust in different environments; simpler alternative below:
region = st.sidebar.selectbox("🌍 Select Region", ["Africa", "Europe", "Asia", "Americas", "Middle East", "Oceania"])
countries = countries_by_region(region)
country = st.sidebar.selectbox("🏳️ Select Country", countries)
model_choice = st.sidebar.radio("🔀 Model", ["Random Forest", "Prophet", "LSTM"])
horizon = st.sidebar.slider("⏳ Forecast horizon (days)", min_value=7, max_value=90, value=30, step=1)

# Data load
DATA_PATH = "data/nasa_power_data_all_params.csv"
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Data file not found at {DATA_PATH}. Please place your dataset there.")
    st.stop()

st.subheader("📊 Dataset Preview")
st.dataframe(preview_table(df, n=8))

# Map
lat, lon = get_country_coords(country)
st.subheader(f"🗺 Location: {country}")
st.plotly_chart(country_map(lat, lon, country), use_container_width=True)

# Filter by country if your dataset has a country column
df_country = filter_by_country(df, country)

# Load model
MODEL_PATHS = {
    "Random Forest": "models/rf_model.pkl",
    "Prophet": "models/prophet_model.pkl",
    "LSTM": "models/lstm_model.h5",
}
model_path = MODEL_PATHS.get(model_choice)
try:
    if model_choice == "Random Forest":
        model = load_rf_model(model_path)
    elif model_choice == "Prophet":
        model = load_prophet_model(model_path)
    else:
        model = load_lstm_model(model_path)
except Exception as e:
    st.error(f"Error loading model for {model_choice}: {e}")
    st.stop()

# Forecast / predict
try:
    out = make_forecast(model, model_choice, df_country, horizon=horizon, date_col=DEFAULT_DATE_COL, target_col=DEFAULT_TARGET_COL)
except Exception as e:
    st.error(f"Error during forecasting: {e}")
    st.stop()

# Display results
st.subheader("📈 Forecast / Prediction Results")

if out.get("forecast_df") is not None:
    # Prophet forecast
    fig = prophet_forecast_plot(out["forecast_df"])
    st.plotly_chart(fig, use_container_width=True)
    st.info("Prophet forecast shown above. Prophet returns future horizon forecasts.")
else:
    y_true = out.get("y_true")
    y_pred = out.get("y_pred")
    if y_true is None or y_pred is None:
        st.warning("Model did not return predictions for the test split.")
    else:
        fig = line_actual_vs_pred(y_true, y_pred, title=f"{model_choice} - Actual vs Predicted (holdout test)")
        st.plotly_chart(fig, use_container_width=True)
        metrics = regression_metrics(y_true, y_pred)
        st.write("### Metrics")
        st.write(f"MAE: {metrics['MAE']:.4f}")
        st.write(f"RMSE: {metrics['RMSE']:.4f}")
        st.write(f"MAPE: {metrics['MAPE']:.2f}%")

# Model comparison (attempt to load all models and compare predictions on dataset)
st.subheader("🔍 Model Comparison (holdout predictions)")

comparison_series = {}
for mname, mpath in MODEL_PATHS.items():
    try:
        if mname == "Random Forest":
            m = load_rf_model(mpath)
            outm = make_forecast(m, "random forest", df_country)
        elif mname == "Prophet":
            m = load_prophet_model(mpath)
            outm = make_forecast(m, "prophet", df_country, horizon=horizon)
        else:
            m = load_lstm_model(mpath)
            outm = make_forecast(m, "lstm", df_country)
    except Exception:
        continue
    if outm.get("y_pred") is not None:
        comparison_series[mname] = outm["y_pred"]

if comparison_series:
    figc = model_comparison_plot(comparison_series, title="Model Predictions Comparison")
    st.plotly_chart(figc, use_container_width=True)
else:
    st.info("No model predictions available for comparison (models may be missing or failed to produce predictions).")

# Footer
st.markdown("---")
st.markdown("Project by Adebowale Immanuel Adeyemi • [GitHub repository](https://github.com/adebowale-aienergy/solar_energy_forecasting_and_dashboard_monitoring)")
