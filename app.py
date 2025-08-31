import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ===============================
# Load Dataset
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("nasa_power_data_all_params.csv")
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df

df = load_data()

# ===============================
# Region & Country Grouping
# ===============================
countries_by_region = {
    "Africa": ["Nigeria", "South Africa", "Kenya", "Egypt", "Morocco"],
    "Europe": ["Germany", "UK", "France", "Spain", "Norway"],
    "Asia": ["India", "China", "Japan", "Saudi Arabia", "UAE"],
    "Americas": ["USA", "Brazil", "Canada", "Mexico", "Chile"],
    "Middle East": ["Turkey", "Iran", "Israel", "Qatar"],
    "Oceania": ["Australia", "New Zealand"]
}

# Example financial defaults ($/MW)
financial_defaults = {
    "Nigeria": {"capex": 0.8, "tariff": 0.12},
    "South Africa": {"capex": 1.0, "tariff": 0.10},
    "Germany": {"capex": 1.2, "tariff": 0.18},
    "USA": {"capex": 1.1, "tariff": 0.15},
    "India": {"capex": 0.7, "tariff": 0.08},
    "China": {"capex": 0.6, "tariff": 0.07},
    "Brazil": {"capex": 0.9, "tariff": 0.11},
    "Australia": {"capex": 1.0, "tariff": 0.14},
}

# ===============================
# Utility Functions
# ===============================
def get_country_center(country_name):
    geolocator = Nominatim(user_agent="solar_app")
    location = geolocator.geocode(country_name)
    if location:
        return location.latitude, location.longitude
    return 0, 0

def train_prophet(ts_data):
    df_train = ts_data.rename(columns={"DATE": "ds", "ALLSKY_KT": "y"})
    model = Prophet()
    model.fit(df_train)
    return model

def train_lstm(ts_data, forecast_horizon):
    data = ts_data["ALLSKY_KT"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(30, len(data_scaled)):
        X.append(data_scaled[i-30:i, 0])
        y.append(data_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # Forecast
    last_30 = data_scaled[-30:]
    forecast = []
    current_input = last_30.reshape((1, 30, 1))

    for _ in range(forecast_horizon * 30):
        next_val = model.predict(current_input, verbose=0)[0][0]
        forecast.append(next_val)
        current_input = np.append(current_input[:, 1:, :], [[[next_val]]], axis=1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    return forecast

def generate_map(selected_country, forecast_horizon):
    solar_map = folium.Map(location=[0, 10], zoom_start=2)

    for country, values in financial_defaults.items():
        avg_generation = 1500 + forecast_horizon * 10
        co2_savings = avg_generation * 0.0007
        capex = values["capex"]
        tariff = values["tariff"]

        revenue = avg_generation * tariff
        payback = (capex * 1_000_000) / revenue if revenue > 0 else None
        roi = (revenue - (capex * 0.05)) / (capex * 1_000_000) * 100

        color = "green" if roi > 10 else "orange" if roi > 5 else "red"
        lat, lon = get_country_center(country)

        folium.CircleMarker(
            location=[lat, lon],
            radius=6 + (co2_savings / 500),
            color=color,
            fill=True,
            fill_color=color,
            popup=folium.Popup(
                f"<b>{country}</b><br>"
                f"CAPEX: ${capex}M/MW<br>"
                f"Tariff: ${tariff}/kWh<br>"
                f"ROI: {roi:.1f}%<br>"
                f"Payback: {payback:.1f} yrs<br>"
                f"CO₂ Savings: {co2_savings:.1f} tons/MW",
                max_width=300
            )
        ).add_to(solar_map)

    return solar_map

# ===============================
# Streamlit App Layout
# ===============================
st.set_page_config(page_title="🌞 Solar Energy Forecast Dashboard", layout="wide")
st.title("🌞 Solar Energy Forecasting & Investment Dashboard")

# Sidebar controls
st.sidebar.header("⚙️ Controls")
region = st.sidebar.selectbox("Select Region", list(countries_by_region.keys()))
country = st.sidebar.selectbox("Select Country", countries_by_region[region])
forecast_horizon = st.sidebar.slider("Forecast Horizon (months)", 1, 12, 6)
model_choice = st.sidebar.radio("Select Forecast Model", ["Random Forest", "Prophet", "LSTM"])

# Filter dataset
df_country = df.copy()

# ===============================
# Tabs Layout
# ===============================
tab1, tab2, tab3 = st.tabs(["📈 Forecast", "💰 Financials", "🌍 Map"])

# ---- Tab 1: Forecast ----
with tab1:
    if model_choice == "Prophet":
        prophet_model = train_prophet(df_country[["DATE", "ALLSKY_KT"]])
        future = prophet_model.make_future_dataframe(periods=forecast_horizon*30)
        forecast = prophet_model.predict(future)

        st.subheader(f"📈 {country} Solar Forecast ({model_choice})")
        fig1, ax = plt.subplots(figsize=(10, 4))
        prophet_model.plot(forecast, ax=ax)
        st.pyplot(fig1)

    elif model_choice == "Random Forest":
        st.subheader(f"📈 {country} Solar Forecast ({model_choice})")
        df_country["Forecast"] = df_country["ALLSKY_KT"].rolling(window=3).mean()
        st.line_chart(df_country.set_index("DATE")[["ALLSKY_KT", "Forecast"]])

    elif model_choice == "LSTM":
        st.subheader(f"📈 {country} Solar Forecast ({model_choice})")
        forecast = train_lstm(df_country, forecast_horizon)

        future_dates = pd.date_range(df_country["DATE"].iloc[-1], periods=len(forecast)+1, freq="D")[1:]
        df_forecast = pd.DataFrame({"DATE": future_dates, "Forecast": forecast.flatten()})

        fig2, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_country["DATE"], df_country["ALLSKY_KT"], label="Actual")
        ax.plot(df_forecast["DATE"], df_forecast["Forecast"], label="LSTM Forecast")
        ax.legend()
        st.pyplot(fig2)

# ---- Tab 2: Financials ----
with tab2:
    if country in financial_defaults:
        capex = financial_defaults[country]["capex"]
        tariff = financial_defaults[country]["tariff"]

        avg_generation = 1500 + forecast_horizon * 10
        revenue = avg_generation * tariff
        payback = (capex * 1_000_000) / revenue if revenue > 0 else None
        roi = (revenue - (capex * 0.05)) / (capex * 1_000_000) * 100
        co2_savings = avg_generation * 0.0007

        st.subheader(f"💰 Financial & Environmental Metrics - {country}")
        c1, c2, c3 = st.columns(3)
        c1.metric("CAPEX", f"${capex}M/MW")
        c2.metric("Tariff", f"${tariff}/kWh")
        c3.metric("ROI", f"{roi:.2f}%")

        c4, c5 = st.columns(2)
        c4.metric("Payback Period", f"{payback:.1f} years")
        c5.metric("CO₂ Savings", f"{co2_savings:.1f} tons/MW")

# ---- Tab 3: Map ----
with tab3:
    st.subheader("🌍 Global Solar Economics & CO₂ Map")
    solar_map = generate_map(country, forecast_horizon)
    st_folium(solar_map, width=900, height=550)
