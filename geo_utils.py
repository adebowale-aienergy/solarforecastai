import pycountry

# Sample mapping (you can expand later with real lat/long or GeoJSON)
COUNTRY_COORDS = {
    "Nigeria": {"lat": 9.0820, "lon": 8.6753},
    "Ghana": {"lat": 7.9465, "lon": -1.0232},
    "Kenya": {"lat": -1.2921, "lon": 36.8219},
    "South Africa": {"lat": -30.5595, "lon": 22.9375},
    "Egypt": {"lat": 26.8206, "lon": 30.8025},
    "United States": {"lat": 37.0902, "lon": -95.7129},
    "Germany": {"lat": 51.1657, "lon": 10.4515},
    "India": {"lat": 20.5937, "lon": 78.9629},
    "China": {"lat": 35.8617, "lon": 104.1954},
    "Brazil": {"lat": -14.2350, "lon": -51.9253}
}

def get_country_coords(country: str):
    """Return (lat, lon) of a country if available, else None."""
    coords = COUNTRY_COORDS.get(country)
    if coords:
        return coords["lat"], coords["lon"]
    return None
