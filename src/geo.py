import pycountry

# Define regions and the countries that belong to them
REGION_COUNTRIES = {
    "Africa": [
        "Nigeria", "Kenya", "South Africa", "Ghana", "Egypt", "Ethiopia",
        "Morocco", "Algeria", "Uganda", "Tanzania"
    ],
    "Europe": [
        "United Kingdom", "Germany", "France", "Spain", "Italy", "Norway",
        "Sweden", "Netherlands", "Poland", "Greece"
    ],
    "Asia": [
        "China", "India", "Japan", "South Korea", "Indonesia", "Saudi Arabia",
        "United Arab Emirates", "Pakistan", "Malaysia", "Bangladesh"
    ],
    "Americas": [
        "United States", "Canada", "Brazil", "Mexico", "Argentina", "Chile",
        "Colombia", "Peru"
    ],
    "Oceania": [
        "Australia", "New Zealand", "Fiji", "Papua New Guinea"
    ],
    "Middle East": [
        "Turkey", "Israel", "Qatar", "Kuwait", "Jordan", "Oman", "Bahrain"
    ]
}


def get_country_regions(available_countries):
    """
    Map available countries (from dataset) into their regions.
    Returns a dict: {region: [countries]}
    """
    regions = {}
    for region, countries in REGION_COUNTRIES.items():
        # Only include countries that exist in the dataset
        region_countries = [c for c in countries if c in available_countries]
        if region_countries:
            regions[region] = sorted(region_countries)
    return regions


def get_country_coordinates(country_name):
    """
    Return (lat, lon) for the given country using pycountry.
    If not found, return (0,0).
    """
    # For now, we keep it simple (could integrate with geopy later)
    COORDS = {
        "Nigeria": (9.082, 8.6753),
        "Kenya": (1.2921, 36.8219),
        "South Africa": (-30.5595, 22.9375),
        "United States": (37.0902, -95.7129),
        "United Kingdom": (55.3781, -3.4360),
        "Germany": (51.1657, 10.4515),
        "China": (35.8617, 104.1954),
        "India": (20.5937, 78.9629),
        "Brazil": (-14.2350, -51.9253),
        "Australia": (-25.2744, 133.7751),
    }
    return COORDS.get(country_name, (0, 0))
