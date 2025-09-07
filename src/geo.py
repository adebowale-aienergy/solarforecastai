import pycountry

def get_country_regions(countries):
    """Group countries into regions (simple fallback)."""
    regions = {
        "Africa": [],
        "Europe": [],
        "Asia": [],
        "Americas": [],
        "Oceania": [],
        "Other": []
    }
    for c in countries:
        try:
            country = pycountry.countries.lookup(c)
            if country.alpha_2 in ["NG", "GH", "ZA", "KE"]:
                regions["Africa"].append(c)
            elif country.alpha_2 in ["US", "CA", "BR", "AR"]:
                regions["Americas"].append(c)
            elif country.alpha_2 in ["CN", "IN", "JP", "ID"]:
                regions["Asia"].append(c)
            elif country.alpha_2 in ["GB", "DE", "FR", "IT"]:
                regions["Europe"].append(c)
            elif country.alpha_2 in ["AU", "NZ"]:
                regions["Oceania"].append(c)
            else:
                regions["Other"].append(c)
        except:
            regions["Other"].append(c)
    return regions

def get_country_coordinates(country_name):
    # Simplified mapping – extend with real lat/lon if needed
    coords = {
        "Nigeria": (9.082, 8.6753),
        "Ghana": (7.9465, -1.0232),
        "Kenya": (-1.2921, 36.8219),
        "United States": (37.0902, -95.7129),
        "India": (20.5937, 78.9629)
    }
    return coords.get(country_name, (0, 0))
