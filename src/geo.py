# src/geo.py
import pycountry

def get_country_regions():
    """Return a dictionary mapping regions to country names."""
    regions = {
        "Africa": [],
        "Europe": [],
        "Asia": [],
        "Americas": [],
        "Oceania": [],
        "Middle East": []
    }

    # Mapping of ISO region keywords → app regions
    region_map = {
        "AF": "Africa",
        "EU": "Europe",
        "AS": "Asia",
        "NA": "Americas",
        "SA": "Americas",
        "OC": "Oceania"
    }

    # Loop through all countries from pycountry
    for country in pycountry.countries:
        # Some countries have alpha_2 codes that map to regions
        try:
            continent_code = country.alpha_2
            # Assign based on first 2 letters of UN M49 code
            # For simplicity, assign Middle East manually
            if country.name in ["United Arab Emirates", "Saudi Arabia", "Qatar", "Israel", "Jordan", "Oman", "Kuwait"]:
                regions["Middle East"].append(country.name)
            else:
                # Default → put into larger buckets
                if continent_code in ["DZ", "NG", "ZA", "KE", "GH", "EG"]:
                    regions["Africa"].append(country.name)
                elif continent_code in ["DE", "FR", "GB", "NO", "ES", "IT"]:
                    regions["Europe"].append(country.name)
                elif continent_code in ["CN", "IN", "JP", "ID", "PK"]:
                    regions["Asia"].append(country.name)
                elif continent_code in ["US", "BR", "CA", "MX", "AR"]:
                    regions["Americas"].append(country.name)
                elif continent_code in ["AU", "NZ", "FJ"]:
                    regions["Oceania"].append(country.name)
        except Exception:
            continue

    return regions


def get_country_coordinates(country):
    """Return approximate (lat, lon) for selected country."""
    coords = {
        "Nigeria": (9.082, 8.6753),
        "Kenya": (-0.0236, 37.9062),
        "South Africa": (-30.5595, 22.9375),
        "Egypt": (26.8206, 30.8025),
        "Ghana": (7.9465, -1.0232),
        "Germany": (51.1657, 10.4515),
        "France": (46.6034, 1.8883),
        "United Kingdom": (55.3781, -3.4360),
        "Norway": (60.4720, 8.4689),
        "Spain": (40.4637, -3.7492),
        "China": (35.8617, 104.1954),
        "India": (20.5937, 78.9629),
        "Japan": (36.2048, 138.2529),
        "Saudi Arabia": (23.8859, 45.0792),
        "United States": (37.0902, -95.7129),
        "Brazil": (-14.2350, -51.9253),
        "Canada": (56.1304, -106.3468),
        "Mexico": (23.6345, -102.5528),
        "Australia": (-25.2744, 133.7751),
        "New Zealand": (-40.9006, 174.8860),
        "United Arab Emirates": (23.4241, 53.8478),
        "Qatar": (25.3548, 51.1839),
        "Israel": (31.0461, 34.8516)
    }
    return coords.get(country, (0, 0))
