from __future__ import annotations
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np
from .constants import DEFAULT_DATE_COL, DEFAULT_TARGET_COL, DEFAULT_COUNTRY_COL

def load_data(filepath: str, parse_dates: bool = True, date_col: str = DEFAULT_DATE_COL, drop_na: bool = True) -> pd.DataFrame:
    """Load CSV file, optionally parse date column and drop NA rows."""
    df = pd.read_csv(filepath)
    if parse_dates and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if drop_na:
        df = df.dropna().reset_index(drop=True)
    return df

def filter_by_country(df: pd.DataFrame, country: str, country_col: str = DEFAULT_COUNTRY_COL) -> pd.DataFrame:
    """Filter dataframe by a country column if present."""
    if country_col in df.columns:
        return df[df[country_col].astype(str).str.lower() == str(country).lower()].reset_index(drop=True)
    return df.copy()

def add_time_features(df: pd.DataFrame, date_col: str = DEFAULT_DATE_COL, include_cyclical: bool = True) -> pd.DataFrame:
    """Add year, month, day, doy, dow and optional cyclical encodings."""
    if date_col not in df.columns:
        return df
    d = pd.to_datetime(df[date_col], errors="coerce")
    df = df.copy()
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["doy"] = d.dt.dayofyear
    df["dow"] = d.dt.weekday
    if include_cyclical:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    return df

def split_features_target(df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COL, drop_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Return X (features) and y (target). drop_cols are removed from X."""
    if drop_cols is None:
        drop_cols = []
    X = df.drop(columns=[c for c in [target_col] + drop_cols if c in df.columns], errors="ignore")
    y = df[target_col] if target_col in df.columns else pd.Series([], dtype=float)
    return X, y

def make_prophet_frame(df: pd.DataFrame, date_col: str = DEFAULT_DATE_COL, target_col: str = DEFAULT_TARGET_COL) -> pd.DataFrame:
    """Return df with columns ds (datetime) and y (target) for Prophet."""
    if date_col not in df.columns or target_col not in df.columns:
        raise ValueError("date or target column not present in dataframe")
    out = df[[date_col, target_col]].copy()
    out.columns = ["ds", "y"]
    out["ds"] = pd.to_datetime(out["ds"])
    return out
