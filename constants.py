from typing import Dict, List, Tuple

# Minimal, editable lists. Add more as you scale.
REGION_COUNTRIES: Dict[str, List[str]] = {
    "Africa": ["Nigeria", "Ghana", "Kenya", "South Africa", "Egypt", "Morocco"],
    "Europe": ["United Kingdom", "Germany", "France", "Norway", "Spain", "Italy"],
    "Asia": ["India", "China", "Japan", "Saudi Arabia", "United Arab Emirates"],
    "Americas": ["United States", "Canada", "Brazil", "Mexico", "Argentina"],
    "Middle East": ["Saudi Arabia", "United Arab Emirates", "Qatar", "Oman"],
    "Oceania": ["Australia", "New Zealand"],
}

# Approximate country centers. Extend as needed.
COUNTRY_COORDS: Dict[str, Tuple[float, float]] = {
    "Nigeria": (9.0820, 8.6753),
    "Ghana": (7.9465, -1.0232),
    "Kenya": (-0.0236, 37.9062),
    "South Africa": (-30.5595, 22.9375),
    "Egypt": (26.8206, 30.8025),
    "Morocco": (31.7917, -7.0926),
    "United Kingdom": (55.3781, -3.4360),
    "Germany": (51.1657, 10.4515),
    "France": (46.2276, 2.2137),
    "Norway": (60.4720, 8.4689),
    "Spain": (40.4637, -3.7492),
    "Italy": (41.8719, 12.5674),
    "India": (20.5937, 78.9629),
    "China": (35.8617, 104.1954),
    "Japan": (36.2048, 138.2529),
    "Saudi Arabia": (23.8859, 45.0792),
    "United Arab Emirates": (23.4241, 53.8478),
    "United States": (37.0902, -95.7129),
    "Canada": (56.1304, -106.3468),
    "Brazil": (-14.2350, -51.9253),
    "Mexico": (23.6345, -102.5528),
    "Argentina": (-38.4161, -63.6167),
    "Qatar": (25.3548, 51.1839),
    "Oman": (21.4735, 55.9754),
    "Australia": (-25.2744, 133.7751),
    "New Zealand": (-40.9006, 174.8860),
}

DEFAULT_TARGET_COL = "TARGET"  # Adjust to your dataset
DEFAULT_DATE_COL = "DATE"      # Adjust to your dataset
