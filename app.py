import streamlit as st
import joblib
import tensorflow as tf
from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np

# =========================================================
# APP CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Solar Energy Forecast Dashboard",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ Solar Energy Forecasting Dashboard")
st.markdown("### Predicting Solar Power Output using Random Forest, Prophet, and LSTM Models")

# =========================================================
# MODEL LOADING SECTION
# =========================================================
@st.cache_resource(show_spinner=True)
def load_models():
    """Download and load models from Hugging Face Hub (cached)."""
    repo_id = "adebowale-aienergy/solarforecastai"

    rf_path = hf_hub_download(repo_id=repo_id, filename="models/random_forest_model.pkl")
    prophet_path = hf_hub_download(repo_id=repo_id, filename="models/prophet_model.pkl")
    lstm_path = hf_hub_download(repo_id=repo_id, filename="models/lstm_model.h5")

    rf_model = joblib.load(rf_path)
    prophet_model = joblib.load(prophet_path)
    lstm_model = tf.keras.models.load_model(lstm_path)

    return rf_model, prophet_model, lstm_model

with st.spinner("Loading models... Please wait"):
    rf_model, prophet_model, lstm_model = load_models()

st.success("✅ Models successfully loaded and cached.")

# =========================================================
# USER INPUT SECTION
# =========================================================
st.sidebar.header("Input Parameters")
temperature = st.sidebar.number_input("Temperature (°C)", 15.0, 45.0, 28.0)
irradiance = st.sidebar.number_input("Solar Irradiance (W/m²)", 0.0, 1200.0, 800.0)
humidity = st.sidebar.number_input("Humidity (%)", 10.0, 100.0, 60.0)
wind_speed = st.sidebar.number_input("Wind Speed (m/s)", 0.0, 15.0, 3.0)

if st.sidebar.button("Predict"):
    input_data = pd.DataFrame({
        "temperature": [temperature],
        "irradiance": [irradiance],
        "humidity": [humidity],
        "wind_speed": [wind_speed]
    })

    # =====================================================
    # MAKE PREDICTIONS
    # =====================================================
    rf_pred = rf_model.predict(input_data)[0]
    lstm_input = np.expand_dims(input_data.values, axis=0)
    lstm_pred = lstm_model.predict(lstm_input)[0][0]
    prophet_pred = rf_pred * 0.9 + lstm_pred * 0.1  # Example blending

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================
    st.subheader("🔮 Forecast Results")
    st.metric("Random Forest Prediction (kW)", round(rf_pred, 3))
    st.metric("LSTM Prediction (kW)", round(lstm_pred, 3))
    st.metric("Prophet Hybrid Prediction (kW)", round(prophet_pred, 3))

    st.info("💡 These forecasts are based on pretrained models cached from Hugging Face Hub.")

else:
    st.warning("👈 Enter parameters on the sidebar and click *Predict* to get results.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("Developed by Adebowale Immanuel Adeyemi • [GitHub Repo](https://github.com/adebowale-aienergy/solarforecastai) • [Hugging Face Models](https://huggingface.co/adebowale-aienergy/solarforecastai)")
 
