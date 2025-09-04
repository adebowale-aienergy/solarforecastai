# app.py

import os
import sys
import streamlit as st
import pandas as pd

# Ensure src is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.constants import (
    DATA_PATH,
    DEFAULT_DATE_COL,
    DEFAULT_TARGET_COL,
    DEFAULT_COUNTRY_COL,
    DEFAULT_HORIZON,
    MIN_HORIZON,
    MAX_HORIZON
)
from src.utils import load_data, preprocess_data, make_forecast
from src.geo import get_country_regions, get_country_coordinates


# ==========================
# Main Streamlit App
# ==========================
def main():
    st.title("☀️ Solar Energy Forecasting Dashboard")

    # Load dataset
    df = load_data(DATA_PATH)

    # Sidebar
    st.sidebar.header("User Input")

    # Country selection
    countries = df[DEFAULT_COUNTRY_COL].unique().tolist()
    country = st.sidebar.selectbox("Select Country", countries)

    # Region info (geo.py groups)
    regions = get_country_regions(countries)
    country_region = None
    for reg, reg_countries in regions.items():
        if country in reg_countries:
            country_region = reg
            break

    st.sidebar.write(f"🌍 Region: {country_region if country_region else 'Unknown'}")

    # Forecast horizon
    horizon = st.sidebar.slider("Forecast Horizon (days)", MIN_HORIZON, MAX_HORIZON, DEFAULT_HORIZON)

    # Preprocess and filter
    df_filtered = preprocess_data(df, country)

    # Show raw data
    with st.expander("🔎 View Raw Data"):
        st.dataframe(df_filtered.head(20))

    # Forecast
    forecast_df = make_forecast(None, df_filtered, horizon)

    st.subheader(f"📈 Forecast for {country} (next {horizon} days)")
    st.line_chart(forecast_df.set_index(DEFAULT_DATE_COL))

    # Show location on map
    lat, lon = get_country_coordinates(country)
    if lat and lon:
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))


if __name__ == "__main__":
    main()
