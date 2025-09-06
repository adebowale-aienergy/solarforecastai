import sys, traceback

print(">>> Streamlit app is starting...")

try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import joblib
except Exception as e:
    print(">>> Import error:", e)
    traceback.print_exc()
    sys.exit(1)

st.set_page_config(page_title="Solar Forecast AI", layout="wide")

st.title("🔆 Solar Forecast AI - Random Forest Debug")
st.write("If you can see this, Random Forest model is being tested.")

# Try loading Random Forest model

try:rf_model = joblib.load("models/random_forest_model.pkl")
    st.success("✅ Random Forest model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading Random Forest model: {e}")
    rf_model = None

# Demo input
if rf_model:
    st.subheader("🌤️ Make a Prediction")
    temp = st.slider("Temperature (°C)", 20, 40, 30)
    irradiance = st.slider("Irradiance (W/m²)", 200, 1000, 600)
    wind = st.slider("Wind Speed (m/s)", 0, 15, 5)

    X = np.array([[temp, irradiance, wind]])
    try:
        prediction = rf_model.predict(X)
        st.success(f"Predicted Solar Power: {prediction[0]:.2f} kW")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
