# src/constants.py

import os

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "nasa_power_data_all_params.csv")

# Model paths
RF_MODEL_PATH = os.path.join(BASE_DIR, "rf_model.pkl")
PROPHET_MODEL_PATH = os.path.join(BASE_DIR, "prophet_model.pkl")
LSTM_MODEL_PATH = os.path.join(BASE_DIR, "lstm_model.h5")

# ----------------------------
# Dataset Defaults
# ----------------------------
# Your dataset columns include: Date, ALLSKY_SFC_SW_DWN, T2M, RH2M, WS2M, PRECTOTCORR, etc.

DEFAULT_DATE_COL = "Date"  # Time column
DEFAULT_TARGET_COL = "ALLSKY_SFC_SW_DWN"  # Solar irradiance (kWh/m²/day)
DEFAULT_COUNTRY_COL = "Country"  # Added dynamically in app.py

# ----------------------------
# Forecast Settings
# ----------------------------
DEFAULT_HORIZON = 7   # Forecast 7 days by default
MIN_HORIZON = 1
MAX_HORIZON = 30

# ----------------------------
# Plotting / Display
# ----------------------------
DATE_FORMAT = "%Y-%m-%d"

