from __future__ import annotations
from typing import Tuple, List
from .constants import REGION_COUNTRIES, COUNTRY_COORDS

def countries_by_region(region: str) -> List[str]:
    """Return list of countries for a region, empty if unknown."""
    return REGION_COUNTRIES.get(region, [])

def get_country_coords(country: str) -> Tuple[float, float]:
    """
    Return (lat, lon) for a country. If unknown, return (0.0, 0.0).
    """
    return COUNTRY_COORDS.get(country, (0.0, 0.0))
