import reverse_geocoder as rg
import pycountry

def get_country_from_coords(lat, lon):
    """Return country code from latitude and longitude using reverse geocoding"""
    try:
        result = rg.search((lat, lon), mode=1)[0]
        return result["cc"]  # e.g. 'NG'
    except Exception:
        return "Unknown"

def country_name_from_code(code):
    """Convert ISO country code to full country name"""
    try:
        return pycountry.countries.get(alpha_2=code).name
    except Exception:
        return code

def add_country_column(df):
    """Ensure dataset has a 'country' column based on lat/lon"""
    if "country" not in df.columns:
        df["country"] = df.apply(lambda row: get_country_from_coords(row["LAT"], row["LON"]), axis=1)
        df["country"] = df["country"].apply(country_name_from_code)
    return df

def get_country_regions(countries):
    """Group countries into regions (example mapping)"""
    regions = {
        "Africa": ["Nigeria", "Ghana", "Kenya", "South Africa"],
        "Europe": ["Germany", "France", "UK", "Norway"],
        "Asia": ["India", "China", "Japan"],
        "Americas": ["USA", "Brazil", "Canada"],
        "Oceania": ["Australia", "New Zealand"],
        "Middle East": ["UAE", "Saudi Arabia", "Israel"]
    }
    return regions
