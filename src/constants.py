# src/constants.py

# Data
# Path to the cleaned and feature-engineered data file
DATA_PATH = "nasa_power_global_32countries_features.csv" # Assuming a new file will be saved with features

# Model paths
# Paths to save the trained models
RF_MODEL_PATH = "rf_solar_forecast_model.pkl"
PROPHET_MODEL_PATH = "prophet_solar_forecast_model.json" # Prophet model is typically saved as JSON
LSTM_MODEL_PATH = "lstm_solar_forecast_model.h5"

# Default column names in the processed data
DATE_COL = "observation_date" # Adjusted to the column name in df_melted/df_features
TARGET_COL = "ALLSKY_SFC_SW_DWN" # The primary target for solar forecasting
COUNTRY_COL = "country" # Adjusted to the column name in df_melted/df_features
PARAMETER_COL = "parameter" # Column containing the climate parameter names
VALUE_COL = "value" # Column containing the parameter values

# Relevant climate parameters for solar forecasting
SOLAR_FORECAST_PARAMETERS = [
    "ALLSKY_SFC_SW_DWN", # Downward Shortwave Radiation at Surface
    "T2M",               # Air Temperature at 2m
    "T2M_MAX",           # Maximum Air Temperature at 2m
    "T2M_MIN",           # Minimum Air Temperature at 2m
    "WS2M",              # Wind Speed at 2m
    "RH2M",              # Relative Humidity at 2m
    "PRECTOTCORR",       # Total Precipitation
    "PS"                 # Surface Pressure
]

# Units for parameters (for display purposes)
PARAMETER_UNITS = {
    "ALLSKY_SFC_SW_DWN": "kWh/m²/day",
    "T2M": "°C",
    "T2M_MAX": "°C",
    "T2M_MIN": "°C",
    "WS2M": "m/s",
    "RH2M": "%",
    "PRECTOTCORR": "mm",
    "PS": "kPa" # Or hPa, depending on the exact unit from the source
}

# Default date range for visualization or initial analysis (optional)
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2024-12-31"

# Key countries/regions of interest for the dashboard (example)
KEY_REGIONS = [
    "USA",
    "India",
    "Germany",
    "Australia",
    "Nigeria" # Example from the list
]

# Forecast horizon (number of days to forecast)
DEFAULT_HORIZON = 7
MIN_HORIZON = 1
MAX_HORIZON = 30

# Features to be used in the models (example, will depend on feature engineering)
MODEL_FEATURES = [
    DATE_COL,
    COUNTRY_COL,
    PARAMETER_COL,
    VALUE_COL,
    "month",
    "year",
    "value_lag1",
    "value_rolling_mean_7d",
    "value_rolling_std_7d",
    "lat", # Include lat/lon as they might be useful features
    "lon"
]

# Features specifically for the Random Forest model (example)
RF_FEATURES = [
    "month",
    "year",
    "value_lag1",
    "value_rolling_mean_7d",
    "value_rolling_std_7d",
    "lat",
    "lon"
    # Potentially encode 'country' or use separate models per country
]

# Features and target for Prophet model (requires 'ds' and 'y')
PROPHET_COLS = {
    'ds': DATE_COL,
    'y': VALUE_COL
}

# Features for LSTM model (will require specific sequence preparation)
LSTM_FEATURES = [
    VALUE_COL,
    "month",
    "year",
    "value_lag1",
    "value_rolling_mean_7d",
    "value_rolling_std_7d",
    "lat",
    "lon"
    # LSTM typically processes sequences of numerical features
]
