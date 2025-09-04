# Country Regions and Coordinates for Dropdown + Map

country_regions = {
    "Africa": ["Nigeria", "Kenya", "South Africa", "Egypt"],
    "Europe": ["Germany", "France", "United Kingdom", "Norway"],
    "Asia": ["India", "China", "Japan"],
    "Americas": ["United States", "Brazil", "Canada"],
    "Middle East": ["Saudi Arabia", "UAE"],
    "Oceania": ["Australia"]
}

country_coords = {
    "Nigeria": (9.082, 8.6753),
    "Kenya": (-0.0236, 37.9062),
    "South Africa": (-30.5595, 22.9375),
    "Egypt": (26.8206, 30.8025),
    "Germany": (51.1657, 10.4515),
    "France": (46.6034, 1.8883),
    "United Kingdom": (55.3781, -3.4360),
    "Norway": (60.4720, 8.4689),
    "India": (20.5937, 78.9629),
    "China": (35.8617, 104.1954),
    "Japan": (36.2048, 138.2529),
    "United States": (37.0902, -95.7129),
    "Brazil": (-14.2350, -51.9253),
    "Canada": (56.1304, -106.3468),
    "Saudi Arabia": (23.8859, 45.0792),
    "UAE": (23.4241, 53.8478),
    "Australia": (-25.2744, 133.7751)
}

def get_country_regions():
    return country_regions

def get_country_coordinates(country):
    return country_coords.get(country, (0, 0))
