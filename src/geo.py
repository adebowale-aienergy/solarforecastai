# src/geo.py

import pandas as pd

# ===========================
# Region Mapping
# ===========================
# You can expand or adjust these groupings if needed
REGION_MAP = {
    "Africa": ["Nigeria", "Kenya", "South Africa", "Ghana", "Egypt"],
    "Europe": ["United Kingdom", "Germany", "France", "Italy", "Spain"],
    "Asia": ["China", "India", "Japan", "South Korea", "Indonesia"],
    "Americas": ["United States", "Canada", "Brazil", "Mexico", "Argentina"],
    "Middle East": ["Saudi Arabia", "UAE", "Iran", "Turkey", "Israel"],
    "Oceania": ["Australia", "New Zealand", "Fiji", "Papua New Guinea"],
}

def get_country_regions(country_list):
    """
    Group countries into regions based on REGION_MAP.
    Returns dict: {region: [countries]}
    """
    regions = {region: [] for region in REGION_MAP.keys()}
    for country in country_list:
        found = False
        for region, countries in REGION_MAP.items():
            if country in countries:
                regions[region].append(country)
                found = True
                break
        if not found:
            # If not found, put in "Other"
            if "Other" not in regions:
                regions["Other"] = []
            regions["Other"].append(country)
    return regions


# ===========================
# Coordinates
# ===========================
def get_country_coordinates(country, dataset_path="nasa_power_data_all_params.csv"):
    """
    Fetch latitude and longitude for a given country from dataset.
    Returns (lat, lon) or (None, None) if not found.
    """
    try:
        df = pd.read_csv(dataset_path)
        if "country" in df.columns and "latitude" in df.columns and "longitude" in df.columns:
            row = df[df["country"] == country].head(1)
            if not row.empty:
                lat = float(row["latitude"].values[0])
                lon = float(row["longitude"].values[0])
                return lat, lon
    except Exception as e:
        print(f"Geo lookup failed: {e}")
    return None, None
