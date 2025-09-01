"""Small general utilities for the app (regions mapping, safe helpers)."""

from typing import List
from .constants import REGION_COUNTRIES

def get_countries_by_region(region: str) -> List[str]:
    """Return list of countries for a region; empty list if not found."""
    return REGION_COUNTRIES.get(region, [])
