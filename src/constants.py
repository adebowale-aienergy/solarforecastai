# src/constants.py

# Data
DATA_PATH = "nasa_power_data_all_params.csv"

# Model paths
RF_MODEL_PATH = "rf_model.pkl"
PROPHET_MODEL_PATH = "prophet_model.pkl"
LSTM_MODEL_PATH = "lstm_model.h5"

# Defaults
DEFAULT_DATE_COL = "DATE"
DEFAULT_TARGET_COL = "TARGET"
DEFAULT_COUNTRY_COL = "COUNTRY"

# Forecast horizon
DEFAULT_HORIZON = 7
MIN_HORIZON = 1
MAX_HORIZON = 30
