"""Geospatial helpers for countries and coordinates."""

from __future__ import annotations
from typing import List, Tuple
from .constants import REGION_COUNTRIES, COUNTRY_COORDS


def countries_by_region(region: str) -> List[str]:
    """Return countries in a region; empty list if region unknown."""
    return REGION_COUNTRIES.get(region, [])


def get_country_coords(country: str) -> Tuple[float, float]:
    """
    Return (lat, lon) for country if found; otherwise (0.0, 0.0).
    """
    return COUNTRY_COORDS.get(country, (0.0, 0.0))
