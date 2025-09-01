"""Data loading and preprocessing utilities."""

from __future__ import annotations
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np

from .constants import DEFAULT_DATE_COL, DEFAULT_TARGET_COL, DEFAULT_COUNTRY_COL


def load_data(filepath: str, parse_dates: bool = True, date_col: str = DEFAULT_DATE_COL, drop_na: bool = True) -> pd.DataFrame:
    """
    Load CSV data into DataFrame.
    - Attempts to parse date_col into datetime if present.
    - Drops NA rows if drop_na True.
    """
    df = pd.read_csv(filepath)
    if parse_dates and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if drop_na:
        df = df.dropna().reset_index(drop=True)
    return df


def ensure_target(df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COL) -> pd.DataFrame:
    """Raise error if target missing, else return df."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe. Columns: {list(df.columns)}")
    return df


def filter_by_country(df: pd.DataFrame, country: str, country_col: str = DEFAULT_COUNTRY_COL) -> pd.DataFrame:
    """
    Filter dataframe by country column (case-insensitive).
    If country_col not present, returns df unchanged.
    """
    if country_col in df.columns:
        mask = df[country_col].astype(str).str.lower() == str(country).lower()
        return df.loc[mask].reset_index(drop=True)
    return df.copy()


def add_time_features(df: pd.DataFrame, date_col: str = DEFAULT_DATE_COL, include_cyclical: bool = True) -> pd.DataFrame:
    """
    Add time-based features: year, month, day, doy, dow, and optional cyclical encodings.
    Returns a new dataframe (copy).
    """
    if date_col not in df.columns:
        return df.copy()
    df2 = df.copy()
    d = pd.to_datetime(df2[date_col], errors="coerce")
    df2["year"] = d.dt.year
    df2["month"] = d.dt.month
    df2["day"] = d.dt.day
    df2["doy"] = d.dt.dayofyear
    df2["dow"] = d.dt.weekday
    if include_cyclical:
        df2["month_sin"] = np.sin(2 * np.pi * df2["month"] / 12)
        df2["month_cos"] = np.cos(2 * np.pi * df2["month"] / 12)
        df2["dow_sin"] = np.sin(2 * np.pi * df2["dow"] / 7)
        df2["dow_cos"] = np.cos(2 * np.pi * df2["dow"] / 7)
    return df2


def split_features_target(df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COL, drop_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split into X (features) and y (target).
    drop_cols: list of columns to drop from features (e.g., id, date, country).
    """
    if drop_cols is None:
        drop_cols = []
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    drop_list = list(set(drop_cols + [target_col]))
    X = df.drop(columns=[c for c in drop_list if c in df.columns], errors="ignore")
    y = df[target_col].copy()
    return X, y


def make_prophet_frame(df: pd.DataFrame, date_col: str = DEFAULT_DATE_COL, target_col: str = DEFAULT_TARGET_COL) -> pd.DataFrame:
    """
    Prepare a DataFrame with columns ['ds','y'] for Prophet.
    Raises ValueError if columns missing.
    """
    if date_col not in df.columns or target_col not in df.columns:
        raise ValueError("make_prophet_frame: date_col or target_col not present in DataFrame.")
    out = df[[date_col, target_col]].copy()
    out.columns = ["ds", "y"]
    out["ds"] = pd.to_datetime(out["ds"])
    return out

