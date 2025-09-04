# ================================
# constants.py
# ================================

# Path to dataset (adjust if using another file location)
DATA_PATH = "data/solar_data.csv"

# Paths to saved models
RF_MODEL_PATH = "models/random_forest.pkl"
PROPHET_MODEL_PATH = "models/prophet_model.pkl"
LSTM_MODEL_PATH = "models/lstm_model.h5"

# Default column names (always uppercase)
DEFAULT_DATE_COL = "DATE"
DEFAULT_TARGET_COL = "TARGET"
DEFAULT_COUNTRY_COL = "COUNTRY"

# Forecast horizon limits
DEFAULT_HORIZON = 7   # 7 days ahead by default
MIN_HORIZON = 1
MAX_HORIZON = 30
