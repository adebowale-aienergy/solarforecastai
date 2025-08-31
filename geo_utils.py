from typing import List, Tuple
from .constants import REGION_COUNTRIES, COUNTRY_COORDS

def countries_by_region(region: str) -> List[str]:
    """Return countries list for a region or empty list."""
    return REGION_COUNTRIES.get(region, [])

def get_country_coords(country: str) -> Tuple[float, float]:
    """Return (lat, lon) for the country or (0.0, 0.0) if unknown."""
    return COUNTRY_COORDS.get(country, (0.0, 0.0))
