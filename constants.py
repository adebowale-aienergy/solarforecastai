"""
Project-wide constants. Adjust column names to match your dataset.
"""
from typing import Dict, Tuple

# Default column names (change to match your CSV if needed)
DEFAULT_DATE_COL = "DATE"       # expected date column name
DEFAULT_TARGET_COL = "TARGET"   # expected solar generation / target column name
DEFAULT_COUNTRY_COL = "COUNTRY" # optional country column

# File paths (relative to project root)
DATA_PATH = "data/nasa_power_data_all_params.csv"
RF_MODEL_PATH = "models/rf_model.pkl"
PROPHET_MODEL_PATH = "models/prophet_model.pkl"
LSTM_MODEL_PATH = "models/lstm_model.h5"

# UI defaults
DEFAULT_HORIZON = 30
MIN_HORIZON = 7
MAX_HORIZON = 90

# Small country list and coords for map centering (extend as required)
REGION_COUNTRIES = {
    "Africa": ["Nigeria", "Ghana", "Kenya", "South Africa", "Egypt", "Morocco"],
    "Europe": ["United Kingdom", "Germany", "France", "Norway", "Spain", "Italy"],
    "Asia": ["India", "China", "Japan", "Saudi Arabia", "United Arab Emirates"],
    "Americas": ["United States", "Canada", "Brazil", "Mexico", "Argentina"],
    "Middle East": ["Saudi Arabia", "United Arab Emirates", "Qatar", "Oman"],
    "Oceania": ["Australia", "New Zealand"],
}

COUNTRY_COORDS: Dict[str, Tuple[float, float]] = {
    "Nigeria": (9.0820, 8.6753),
    "Ghana": (7.9465, -1.0232),
    "Kenya": (-0.0236, 37.9062),
    "South Africa": (-30.5595, 22.9375),
    "Egypt": (26.8206, 30.8025),
    "United Kingdom": (55.3781, -3.4360),
    "Germany": (51.1657, 10.4515),
    "France": (46.2276, 2.2137),
    "India": (20.5937, 78.9629),
    "China": (35.8617, 104.1954),
    "United States": (37.0902, -95.7129),
    "Australia": (-25.2744, 133.7751),
}
