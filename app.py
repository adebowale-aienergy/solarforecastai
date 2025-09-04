import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------
# File paths (adjust if needed)
# ---------------------------
DATA_PATH = "nasa_power_data_all_params.csv"

# ---------------------------
# Utility functions
# ---------------------------
def get_country_regions(countries):
    """Group countries by regions for dropdown selection."""
    regions = {
        "Africa": ["Nigeria", "Kenya", "South Africa", "Egypt", "Ghana"],
        "Europe": ["Germany", "France", "United Kingdom", "Norway", "Spain"],
        "Asia": ["India", "China", "Japan", "Saudi Arabia", "UAE"],
        "Americas": ["United States", "Canada", "Brazil", "Mexico", "Argentina"],
        "Oceania": ["Australia", "New Zealand"],
    }

    # Keep only available countries from dataset
    region_map = {}
    for region, region_countries in regions.items():
        available = [c for c in region_countries if c in countries]
        if available:
            region_map[region] = available

    return region_map


def plot_time_series(df, country, target_col="ALLSKY_KT"):
    """Plot time series of a selected parameter."""
    fig, ax = plt.subplots(figsize=(10, 4))
    subset = df[df["country"] == country]
    ax.plot(pd.to_datetime(subset["date"]), subset[target_col], label=target_col)
    ax.set_title(f"{target_col} over time in {country}")
    ax.set_xlabel("Date")
    ax.set_ylabel(target_col)
    ax.legend()
    st.pyplot(fig)


# ---------------------------
# Main Streamlit App
# ---------------------------
def main():
    st.title("☀️ Solar Energy Forecasting Dashboard")
    st.markdown("Explore and forecast solar energy parameters using NASA POWER dataset.")

    # Load dataset
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Dataset not found at `{DATA_PATH}`. Please upload it.")
        return

    # Ensure required columns exist
    required_cols = ["country", "date"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Dataset is missing required column: `{col}`")
            return

    # Extract countries and regions
    countries = df["country"].dropna().unique().tolist()
    regions = get_country_regions(countries)

    # Sidebar filters
    st.sidebar.header("🌍 Filters")
    region = st.sidebar.selectbox("Select a region", list(regions.keys()))
    country = st.sidebar.selectbox("Select a country", regions[region])
    target_col = st.sidebar.selectbox("Select parameter", [c for c in df.columns if c not in ["country", "date"]])

    # Plot
    st.subheader(f"📊 {target_col} in {country}")
    plot_time_series(df, country, target_col)

    # Show raw data option
    if st.checkbox("Show raw data"):
        st.write(df[df["country"] == country].head())


# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    main()
