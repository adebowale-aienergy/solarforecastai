"""Data loading and preprocessing utilities."""

from __future__ import annotations
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np

# Import constants from src/constants.py
from src.constants import (
    DATE_COL, TARGET_COL, COUNTRY_COL, PARAMETER_COL, VALUE_COL,
    SOLAR_FORECAST_PARAMETERS, RF_FEATURES, LSTM_FEATURES
)


def load_processed_data(filepath: str) -> pd.DataFrame:
    """Load processed CSV data into DataFrame and standardize date column."""
    df = pd.read_csv(filepath)

    # Standardize date column
    for col in ['observation_date', 'date']:
        if col in df.columns:
            df.rename(columns={col: 'ds'}, inplace=True)
            df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
            break

    df.dropna(subset=['ds'], inplace=True)
    return df


def filter_data(
    df: pd.DataFrame,
    country: Optional[str] = None,
    parameter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """Filter dataframe by country, parameter, and date range."""
    filtered_df = df.copy()

    if country:
        filtered_df = filtered_df[filtered_df[COUNTRY_COL].astype(str).str.lower() == str(country).lower()]

    if parameter:
        filtered_df = filtered_df[filtered_df[PARAMETER_COL] == parameter]

    if start_date:
        filtered_df = filtered_df[filtered_df['ds'] >= pd.to_datetime(start_date)]

    if end_date:
        filtered_df = filtered_df[filtered_df['ds'] <= pd.to_datetime(end_date)]

    return filtered_df.reset_index(drop=True)


def prepare_data_for_model(
    df: pd.DataFrame,
    target_parameter: str = TARGET_COL,
    features: List[str] = RF_FEATURES,
    target_col_name: str = VALUE_COL
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features (X) and target (y) for model training."""
    df_filtered = df[df[PARAMETER_COL] == target_parameter].copy()

    X_cols = [col for col in features if col in df_filtered.columns and col not in [target_col_name, PARAMETER_COL, 'ds']]
    y_col = target_col_name

    X = df_filtered[X_cols].copy() if X_cols else pd.DataFrame(index=df_filtered.index)
    y = df_filtered[y_col].copy()

    combined = pd.concat([X, y], axis=1).dropna()
    X = combined[X_cols] if X_cols else pd.DataFrame(index=combined.index)
    y = combined[y_col]

    return X, y


def make_prophet_frame(
    df: pd.DataFrame,
    parameter: str = TARGET_COL,
    target_col: str = VALUE_COL
) -> pd.DataFrame:
    """Prepare DataFrame ['ds','y'] for Prophet."""
    if PARAMETER_COL not in df.columns or 'ds' not in df.columns or target_col not in df.columns:
        raise ValueError("Required columns for Prophet not present.")

    df_filtered = df[df[PARAMETER_COL] == parameter].copy()
    if df_filtered.empty:
        return pd.DataFrame(columns=['ds', 'y'])

    prophet_df = df_filtered[['ds', target_col]].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
    prophet_df.dropna(subset=['y'], inplace=True)

    return prophet_df


def get_unique_values(df: pd.DataFrame, column: str) -> list:
    if column in df.columns:
        return sorted(df[column].unique().tolist())
    return []


def get_solar_parameters() -> List[str]:
    return SOLAR_FORECAST_PARAMETERS
