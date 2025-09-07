import os

# ==== PATHS ====
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "nasa_power_data_all_params.csv")
RF_MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")
PROPHET_MODEL_PATH = os.path.join(BASE_DIR, "models", "prophet_model.pkl")
LSTM_MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_model.h5")

# ==== DEFAULTS ====
DEFAULT_DATE_COL = "date"
DEFAULT_TARGET_COL = "solar_radiation"   # adjust if your dataset uses another name
DEFAULT_COUNTRY_COL = "country"

# Forecast horizon
DEFAULT_HORIZON = 7
MIN_HORIZON = 1
MAX_HORIZON = 30
