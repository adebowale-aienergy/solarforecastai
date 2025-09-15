"""Data loading and preprocessing utilities."""

from __future__ import annotations
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np

# Import constants from the updated constants.py
from constants import (
    DATA_PATH, DATE_COL, TARGET_COL, COUNTRY_COL,
    PARAMETER_COL, VALUE_COL, SOLAR_FORECAST_PARAMETERS,
    PROPHET_COLS, MODEL_FEATURES
)


def load_processed_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the processed CSV data (with features) into DataFrame.
    Assumes the date column is already in a suitable format or can be coerced.
    """
    df = pd.read_csv(filepath)
    # Ensure the date column is in datetime format
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    # Drop rows where the date could not be parsed
    df.dropna(subset=[DATE_COL], inplace=True)
    return df


def filter_data(
    df: pd.DataFrame,
    country: Optional[str] = None,
    parameter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter dataframe by country, parameter, and date range.
    Uses the column names defined in constants.py.
    """
    filtered_df = df.copy()

    if country:
        if COUNTRY_COL in filtered_df.columns:
            mask_country = filtered_df[COUNTRY_COL].astype(str).str.lower() == str(country).lower()
            filtered_df = filtered_df.loc[mask_country]
        else:
            print(f"Warning: Country column '{COUNTRY_COL}' not found for filtering.")


    if parameter:
         if PARAMETER_COL in filtered_df.columns:
             mask_param = filtered_df[PARAMETER_COL] == parameter
             filtered_df = filtered_df.loc[mask_param]
         else:
            print(f"Warning: Parameter column '{PARAMETER_COL}' not found for filtering.")


    if start_date and DATE_COL in filtered_df.columns:
        try:
            start_date_dt = pd.to_datetime(start_date)
            mask_start_date = filtered_df[DATE_COL] >= start_date_dt
            filtered_df = filtered_df.loc[mask_start_date]
        except ValueError:
            print(f"Warning: Invalid start_date format: {start_date}")

    if end_date and DATE_COL in filtered_df.columns:
        try:
            end_date_dt = pd.to_datetime(end_date)
            mask_end_date = filtered_df[DATE_COL] <= end_date_dt
            filtered_df = filtered_df.loc[mask_end_date]
        except ValueError:
            print(f"Warning: Invalid end_date format: {end_date}")

    return filtered_df.reset_index(drop=True)


def prepare_data_for_model(
    df: pd.DataFrame,
    target_parameter: str = TARGET_COL,
    features: List[str] = MODEL_FEATURES,
    target_col_name: str = VALUE_COL
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for model training.
    Filters for the target parameter and splits into features (X) and target (y).
    Assumes features like lagged values and rolling statistics are already created.
    """
    # Filter for the specific parameter that is the target variable
    if PARAMETER_COL in df.columns:
        df_filtered_param = df[df[PARAMETER_COL] == target_parameter].copy()
    else:
        raise ValueError(f"Parameter column '{PARAMETER_COL}' not found in dataframe.")

    # Ensure the target column exists after filtering
    if target_col_name not in df_filtered_param.columns:
         raise ValueError(f"Target value column '{target_col_name}' not found after parameter filtering.")


    # Define features and target based on the filtered data and constants
    # Ensure only relevant features that exist in the dataframe are selected
    X_cols = [col for col in features if col in df_filtered_param.columns and col != target_col_name and col != PARAMETER_COL and col != DATE_COL]
    y_col = target_col_name

    if y_col not in df_filtered_param.columns:
         raise ValueError(f"Target column '{y_col}' not found in the filtered dataframe.")
    if not X_cols:
         print("Warning: No feature columns found based on MODEL_FEATURES and available columns.")
         # Proceed with an empty features dataframe if no valid feature columns are found
         X = pd.DataFrame(index=df_filtered_param.index)
    else:
         X = df_filtered_param[X_cols].copy()

    y = df_filtered_param[y_col].copy()

    # Handle potential NaNs in features or target after filtering/selection
    # Simple imputation (e.g., mean or median) or dropping rows could be done here
    # For now, let's drop rows with NaNs in either X or y
    # Combine X and y to drop corresponding rows
    combined_df = pd.concat([X, y], axis=1)
    combined_df.dropna(inplace=True)

    # Separate X and y again after dropping NaNs
    if not X_cols:
         X = pd.DataFrame(index=combined_df.index) # Recreate empty X with correct index
    else:
        X = combined_df[X_cols].copy()
    y = combined_df[y_col].copy()


    return X, y


def make_prophet_frame(df: pd.DataFrame, date_col: str = DATE_COL, target_col: str = VALUE_COL, parameter: str = TARGET_COL) -> pd.DataFrame:
    """
    Prepare a DataFrame with columns ['ds','y'] for Prophet for a specific parameter.
    Filters for the specified parameter before creating the Prophet frame.
    Raises ValueError if columns missing or parameter not found.
    """
    if date_col not in df.columns or target_col not in df.columns or PARAMETER_COL not in df.columns:
        raise ValueError("make_prophet_frame: Required columns (date, target value, parameter) not present in DataFrame.")

    # Filter for the specific parameter
    df_filtered_param = df[df[PARAMETER_COL] == parameter].copy()

    if df_filtered_param.empty:
         print(f"Warning: No data found for parameter '{parameter}' to create Prophet frame.")
         return pd.DataFrame(columns=['ds', 'y']) # Return empty DataFrame if no data for parameter


    out = df_filtered_param[[date_col, target_col]].copy()
    out.columns = ["ds", "y"]
    out["ds"] = pd.to_datetime(out["ds"])
    # Prophet requires 'y' to be numeric
    out["y"] = pd.to_numeric(out["y"], errors='coerce')
    # Drop rows where 'y' could not be converted to numeric
    out.dropna(subset=['y'], inplace=True)

    return out


# Add a function to get unique values for filtering in the dashboard
def get_unique_values(df: pd.DataFrame, column: str) -> list:
    """
    Get unique values from a specified column in the DataFrame.
    Returns an empty list if the column does not exist.
    """
    if column in df.columns:
        # Convert to list and sort for consistent order
        return sorted(df[column].unique().tolist())
    else:
        print(f"Warning: Column '{column}' not found in the DataFrame.")
        return []

# Add a function to get parameters relevant for solar forecasting
def get_solar_parameters() -> List[str]:
    """
    Returns the list of climate parameters relevant for solar forecasting.
    """
    return SOLAR_FORECAST_PARAMETERS
