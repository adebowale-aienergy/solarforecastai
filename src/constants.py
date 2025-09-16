# src/constants.py

# =========================
# Data and File Paths
# =========================
DATA_PATH = "https://raw.githubusercontent.com/adebowale-aienergy/solarforecastai/refs/heads/main/data/nasa_power_global_32countries.csv"  
# Raw link from GitHub – ensures Streamlit and Colab can load data directly

# Model paths
RF_MODEL_PATH = "rf_solar_forecast_model.pkl"
PROPHET_MODEL_PATH = "prophet_solar_forecast_model.json"  # Prophet models usually saved as JSON
LSTM_MODEL_PATH = "lstm_solar_forecast_model.h5"


# =========================
# Column Names
# =========================
DATE_COL = "observation_date"  # Date column in dataset
TARGET_COL = "ALLSKY_SFC_SW_DWN"  # Primary target for solar forecasting
COUNTRY_COL = "country"          # Country column
PARAMETER_COL = "parameter"      # Climate parameter name
VALUE_COL = "value"              # Parameter values


# =========================
# Climate Parameters
# =========================
SOLAR_FORECAST_PARAMETERS = [
    "ALLSKY_SFC_SW_DWN",  # Downward Shortwave Radiation at Surface
    "T2M",                # Air Temperature at 2m
    "T2M_MAX",            # Maximum Air Temperature at 2m
    "T2M_MIN",            # Minimum Air Temperature at 2m
    "WS2M",               # Wind Speed at 2m
    "RH2M",               # Relative Humidity at 2m
    "PRECTOTCORR",        # Total Precipitation
    "PS"                  # Surface Pressure
]

PARAMETER_UNITS = {
    "ALLSKY_SFC_SW_DWN": "kWh/m²/day",
    "T2M": "°C",
    "T2M_MAX": "°C",
    "T2M_MIN": "°C",
    "WS2M": "m/s",
    "RH2M": "%",
    "PRECTOTCORR": "mm",
    "PS": "kPa"  # Or hPa depending on NASA POWER export
}


# =========================
# Default Settings
# =========================
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2024-12-31"

KEY_REGIONS = [
    "USA",
    "India",
    "Germany",
    "Australia",
    "Nigeria"
]

DEFAULT_HORIZON = 7
MIN_HORIZON = 1
MAX_HORIZON = 30


# =========================
# Model Features
# =========================
# These are the engineered features you’ll actually feed into models
MODEL_FEATURES = [
    "month",
    "year",
    "value_lag1",
    "value_rolling_mean_7d",
    "value_rolling_std_7d",
    "lat",
    "lon"
]

# Random Forest features
RF_FEATURES = MODEL_FEATURES.copy()

# Prophet expects only ['ds', 'y']
PROPHET_COLS = ["ds", "y"]

# LSTM features (numerical only, sequences)
LSTM_FEATURES = [
    VALUE_COL,
    "month",
    "year",
    "value_lag1",
    "value_rolling_mean_7d",
    "value_rolling_std_7d",
    "lat",
    "lon"
]
