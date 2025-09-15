# src/geo.py

def get_country_regions():
    """Returns a dictionary mapping regions to lists of countries."""
    return {
        # Updated to match the 32 countries in the dataset
        "Africa": ["Nigeria", "South Africa", "Egypt", "Kenya", "Morocco", "Ghana"],
        "North America": ["USA", "Canada"], # Updated region name
        "South America": ["Brazil", "Argentina", "Chile", "Colombia", "Peru"], # Added missing countries
        "Europe": ["Germany", "France", "United Kingdom", "Italy", "Spain", "Norway", "Poland", "Sweden"], # Updated UK, added missing countries
        "Asia": ["India", "China", "Japan", "South Korea", "Indonesia"], # Added missing country
        "Middle East": ["Saudi Arabia", "UAE", "Turkey", "Iran"], # Added missing country
        "Oceania": ["Australia", "New Zealand"]
    }

def get_country_coordinates():
    """Returns a dictionary mapping country names to their coordinates [lat, lon]."""
    return {
        # Updated to match the 32 countries in the dataset
        ("Nigeria", 9.0820, 8.6753),
        ("South Africa", -30.5595, 22.9375),
        ("Egypt", 26.8206, 30.8025),
        ("Kenya", -0.0236, 37.9062),
        ("Morocco", 31.7917, -7.0926),
        ("Ghana", 7.9465, -1.0232),
        ("USA", 37.0902, -95.7129),
        ("Canada", 56.1304, -106.3468),
        ("Brazil", -14.2350, -51.9253),
        ("Argentina", -38.4161, -63.6167),
        ("Chile", -35.6751, -71.5430),
        ("Colombia", 4.5709, -74.2973),
        ("Peru", -9.1900, -75.0152),
        ("Germany", 51.1657, 10.4515),
        ("France", 46.6034, 1.8883),
        ("United Kingdom", 55.3781, -3.4360),
        ("Italy", 41.8719, 12.5674),
        ("Spain", 40.4637, -3.7492),
        ("Norway", 60.4720, 8.4689),
        ("Poland", 51.9194, 19.1451),
        ("Sweden", 60.1282, 18.6435),
        ("India", 20.5937, 78.9629),
        ("China", 35.8617, 104.1954),
        ("Japan", 36.2048, 138.2529),
        ("South Korea", 35.9078, 127.7669),
        ("Indonesia", -0.7893, 113.9213),
        ("Saudi Arabia", 23.8859, 45.0792),
        ("UAE", 23.4241, 53.8478),
        ("Turkey", 38.9637, 35.2433),
        ("Iran", 32.4279, 53.6880),
        ("Australia", -25.2744, 133.7751),
        ("New Zealand", -40.9006, 174.8860)
    }
    # Convert the list of tuples back to a dictionary for consistency,
    # using the country name as the key and [lat, lon] as the value.
    return {country: [lat, lon] for country, lat, lon in country_coords_list}


# Re-writing the get_country_coordinates function to return a dictionary directly
def get_country_coordinates():
    """Returns a dictionary mapping country names to their coordinates [lat, lon]."""
    return {
        # Updated to match the 32 countries in the dataset
        "Nigeria": [9.0820, 8.6753],
        "South Africa": [-30.5595, 22.9375],
        "Egypt": [26.8206, 30.8025],
        "Kenya": [-0.0236, 37.9062],
        "Morocco": [31.7917, -7.0926],
        "Ghana": [7.9465, -1.0232],
        "USA": [37.0902, -95.7129],
        "Canada": [56.1304, -106.3468],
        "Brazil": [-14.2350, -51.9253],
        "Argentina": [-38.4161, -63.6167],
        "Chile": [-35.6751, -71.5430],
        "Colombia": [4.5709, -74.2973],
        "Peru": [-9.1900, -75.0152],
        "Germany": [51.1657, 10.4515],
        "France": [46.6034, 1.8883],
        "United Kingdom": [55.3781, -3.4360],
        "Italy": [41.8719, 12.5674],
        "Spain": [40.4637, -3.7492],
        "Norway": [60.4720, 8.4689],
        "Poland": [51.9194, 19.1451],
        "Sweden": [60.1282, 18.6435],
        "India": [20.5937, 78.9629],
        "China": [35.8617, 104.1954],
        "Japan": [36.2048, 138.2529],
        "South Korea": [35.9078, 127.7669],
        "Indonesia": [-0.7893, 113.9213],
        "Saudi Arabia": [23.8859, 45.0792],
        "UAE": [23.4241, 53.8478],
        "Turkey": [38.9637, 35.2433],
        "Iran": [32.4279, 53.6880],
        "Australia": [-25.2744, 133.7751],
        "New Zealand": [-40.9006, 174.8860]
    }

# Potentially add functions for:
# - Getting a list of all countries
# - Getting the region for a given country
# - Calculating distance between coordinates (if needed)
# - (For dashboard) functions that might integrate with a mapping library (though this might be better in a separate viz file)
