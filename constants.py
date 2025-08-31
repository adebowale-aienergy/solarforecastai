from typing import Dict, List, Tuple

# Regions and a small country list to start — extend as needed.
REGION_COUNTRIES: Dict[str, List[str]] = {
    "Africa": ["Nigeria", "Ghana", "Kenya", "South Africa", "Egypt", "Morocco"],
    "Europe": ["United Kingdom", "Germany", "France", "Norway", "Spain", "Italy"],
    "Asia": ["India", "China", "Japan", "Saudi Arabia", "United Arab Emirates"],
    "Americas": ["United States", "Canada", "Brazil", "Mexico", "Argentina"],
    "Middle East": ["Saudi Arabia", "United Arab Emirates", "Qatar", "Oman"],
    "Oceania": ["Australia", "New Zealand"],
}

# Approximate country centers (lat, lon)
COUNTRY_COORDS: Dict[str, Tuple[float, float]] = {
    "Nigeria": (9.0820, 8.6753),
    "Ghana": (7.9465, -1.0232),
    "Kenya": (-0.0236, 37.9062),
    "South Africa": (-30.5595, 22.9375),
    "Egypt": (26.8206, 30.8025),
    "United States": (37.0902, -95.7129),
    "India": (20.5937, 78.9629),
    "Australia": (-25.2744, 133.7751),
}

# Default column names — change if your CSV uses different names
DEFAULT_TARGET_COL = "TARGET"
DEFAULT_DATE_COL = "DATE"
DEFAULT_COUNTRY_COL = "COUNTRY"
