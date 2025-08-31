from __future__ import annotations
from typing import Tuple, Optional, Dict
import pandas as pd
import numpy as np

from .constants import DEFAULT_TARGET_COL, DEFAULT_DATE_COL

def load_data(
    filepath: str,
    parse_dates: bool = True,
    date_col: str = DEFAULT_DATE_COL,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Load CSV, optionally parse date column, and clean NAs.
    """
    df = pd.read_csv(filepath)
    if parse_dates and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if drop_na:
        df = df.dropna()
    return df

def rename_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Rename columns using a mapping dict."""
    return df.rename(columns=mapping)

def filter_by_country(df: pd.DataFrame, country_col: str, country: str) -> pd.DataFrame:
    """Return rows for a single country if present; else original df."""
    if country_col in df.columns and country in df[country_col].unique():
        return df[df[country_col] == country].copy()
    return df.copy()

def add_time_features(
    df: pd.DataFrame,
    date_col: str = DEFAULT_DATE_COL,
    include_cyclical: bool = True,
) -> pd.DataFrame:
    """
    Add common time features: year, month, day, doy, dow.
    Optionally add cyclical encodings for month and day of week.
    """
    if date_col not in df.columns:
        return df

    d = df[date_col]
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

def split_features_target(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COL,
    drop_cols: Optional[list[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split into X, y. Optionally drop non-feature columns (e.g., ids/text/date).
    """
    if drop_cols:
        X = df.drop(columns=list(set(drop_cols + [target_col])), errors="ignore")
    else:
        X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col]
    return X, y

def make_prophet_frame(
    df: pd.DataFrame,
    date_col: str = DEFAULT_DATE_COL,
    target_col: str = DEFAULT_TARGET_COL,
) -> pd.DataFrame:
    """
    Prepare dataframe for Prophet: columns 'ds' (datetime) and 'y' (target).
    """
    if date_col not in df.columns or target_col not in df.columns:
        raise ValueError("Dataframe must contain date and target columns.")
    out = df[[date_col, target_col]].copy()
    out.columns = ["ds", "y"]
    return out
