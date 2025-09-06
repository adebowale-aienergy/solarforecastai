# app.py (debug version)

import sys, traceback

print(">>> Streamlit app is starting...")

try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import joblib
    import tensorflow as tf
    from prophet import Prophet

    print(">>> All imports successful!")

except Exception as e:
    print(">>> Import error:", e)
    traceback.print_exc()
    sys.exit(1)

# Basic Streamlit test UI
st.set_page_config(page_title="Solar Forecast AI (Debug Mode)", layout="wide")

st.title("🔆 Solar Forecast AI - Debug Mode")
st.write("If you can see this, the app is running.")

# Simple test DataFrame
df = pd.DataFrame({
    "time": pd.date_range("2025-01-01", periods=10, freq="D"),
    "solar": np.random.rand(10)
})
st.line_chart(df.set_index("time"))

st.success("✅ Debug app loaded successfully!")
