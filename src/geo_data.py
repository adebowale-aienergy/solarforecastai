# geo_data.py

# Dictionary of countries with approximate central coordinates
COUNTRY_COORDS = {
    # --- Africa ---
    "Nigeria": {"lat": 9.082, "lon": 8.675},
    "South Africa": {"lat": -30.559, "lon": 22.937},
    "Kenya": {"lat": -0.0236, "lon": 37.9062},
    "Egypt": {"lat": 26.8206, "lon": 30.8025},
    "Morocco": {"lat": 31.7917, "lon": -7.0926},
    
    # --- Europe ---
    "Germany": {"lat": 51.1657, "lon": 10.4515},
    "France": {"lat": 46.6034, "lon": 1.8883},
    "United Kingdom": {"lat": 55.3781, "lon": -3.4360},
    "Spain": {"lat": 40.4637, "lon": -3.7492},
    "Italy": {"lat": 41.8719, "lon": 12.5674},

    # --- Asia ---
    "China": {"lat": 35.8617, "lon": 104.1954},
    "India": {"lat": 20.5937, "lon": 78.9629},
    "Japan": {"lat": 36.2048, "lon": 138.2529},
    "South Korea": {"lat": 35.9078, "lon": 127.7669},
    "Saudi Arabia": {"lat": 23.8859, "lon": 45.0792},

    # --- Americas ---
    "United States": {"lat": 37.0902, "lon": -95.7129},
    "Brazil": {"lat": -14.2350, "lon": -51.9253},
    "Mexico": {"lat": 23.6345, "lon": -102.5528},
    "Canada": {"lat": 56.1304, "lon": -106.3468},
    "Argentina": {"lat": -38.4161, "lon": -63.6167},

    # --- Middle East ---
    "UAE": {"lat": 23.4241, "lon": 53.8478},
    "Israel": {"lat": 31.0461, "lon": 34.8516},
    "Qatar": {"lat": 25.3548, "lon": 51.1839},
    "Iran": {"lat": 32.4279, "lon": 53.6880},
    "Turkey": {"lat": 38.9637, "lon": 35.2433},

    # --- Oceania ---
    "Australia": {"lat": -25.2744, "lon": 133.7751},
    "New Zealand": {"lat": -40.9006, "lon": 174.8860},
    "Fiji": {"lat": -17.7134, "lon": 178.0650},
    "Papua New Guinea": {"lat": -6.314993, "lon": 143.95555},
    "Samoa": {"lat": -13.759, "lon": -172.1046},
}

def attach_geo_data(df, country_col="country"):
    """
    Adds lat/lon to a DataFrame based on country name.
    """
    df["lat"] = df[country_col].map(lambda x: COUNTRY_COORDS.get(x, {}).get("lat"))
    df["lon"] = df[country_col].map(lambda x: COUNTRY_COORDS.get(x, {}).get("lon"))
    return df
