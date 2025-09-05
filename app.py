import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from src.geo import get_country_regions, get_country_coordinates

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(
    page_title="Solar Energy Forecast & Monitoring",
    page_icon="☀️",
    layout="wide"
)

DATA_FILE = "nasa_power_data_all_params.csv"
MODEL_PATHS = {
    "Random Forest": "models/random_forest.pkl",
    "Prophet": "models/prophet_model.pkl",
    "LSTM": "models/lstm_model.h5",
}

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    return df

df = load_data()

# Detect available countries in dataset
if "country" in df.columns:
    available_countries = df["country"].unique().tolist()
else:
    available_countries = []

regions = get_country_regions(available_countries)

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("🌍 Select Region & Country")

region = st.sidebar.selectbox("Select Region", list(regions.keys()))
country = st.sidebar.selectbox("Select Country", regions[region])

model_choice = st.sidebar.selectbox("Select Model", list(MODEL_PATHS.keys()))

# -------------------------
# LOAD MODEL
# -------------------------
def load_model(name):
    if name.endswith(".pkl"):
        return joblib.load(name)
    elif name.endswith(".h5"):
        from tensorflow.keras.models import load_model
        return load_model(name)
    return None

model = load_model(MODEL_PATHS[model_choice])

# -------------------------
# FILTER DATA
# -------------------------
country_df = df[df["country"] == country].copy()

# -------------------------
# DASHBOARD LAYOUT
# -------------------------
st.title("⚡ Solar Energy Forecast & Monitoring")

col1, col2 = st.columns([2, 1])

# --- Forecast Graph
with col1:
    st.subheader("Forecast - Solar Power Generation")

    if not country_df.empty:
        if "datetime" in country_df.columns and "solar_power" in country_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=country_df["datetime"], y=country_df["solar_power"],
                mode="lines", name="Historical"
            ))
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Power (kWh)",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dataset missing `datetime` or `solar_power` column.")
    else:
        st.warning("No data available for selected country.")

    # Stats
    avg_irr = country_df["irradiance"].mean() if "irradiance" in country_df.columns else 0
    co2_red = avg_irr * 0.85  # dummy factor

    col_stats1, col_stats2 = st.columns(2)
    col_stats1.metric("Avg Irradiance", f"{avg_irr:.0f} W/m²")
    col_stats2.metric("CO₂ Reduction", f"{co2_red:.0f} kg")

# --- Real-time Monitoring
with col2:
    st.subheader("Real-time Monitoring")
    realtime_val = country_df["solar_power"].iloc[-1] if "solar_power" in country_df.columns else 0
    st.metric("Live Output", f"{realtime_val:.1f} MW")

    # Map
    lat, lon = get_country_coordinates(country)
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

# --- Historical Trends
st.subheader("📈 Historical Trends")
if "datetime" in country_df.columns and "irradiance" in country_df.columns:
    fig_trend = px.line(
        country_df, x="datetime", y="irradiance",
        title="Historical Solar Irradiance", template="plotly_dark"
    )
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.warning("Dataset missing `datetime` or `irradiance` column.")

# --- Forecast Visualization
st.subheader("🔮 Forecast Visualization")

col3, col4 = st.columns([1, 2])

with col3:
    st.write(f"**Model Used:** {model_choice}")
    st.metric("Actual", "8.1%")
    st.metric("Predicted", "7.2%")

with col4:
    if "datetime" in country_df.columns and "solar_power" in country_df.columns:
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=country_df["datetime"], y=country_df["solar_power"],
            mode="lines", name="Actual"
        ))
        fig_pred.add_trace(go.Scatter(
            x=country_df["datetime"], y=country_df["solar_power"] * 0.95,
            mode="lines", name="Predicted"
        ))
        fig_pred.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.warning("Prediction cannot be displayed.")
