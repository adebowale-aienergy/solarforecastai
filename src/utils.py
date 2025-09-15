"""General utility functions for the project."""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
# Add other necessary imports for general utilities if needed (e.g., logging)

# Import constants if any general utility function needs them
from constants import DATA_PATH, DATE_COL, COUNTRY_COL, PARAMETER_COL, VALUE_COL


# Example of a general utility function: Check if a file exists
def file_exists(filepath: str) -> bool:
    """Checks if a file exists at the given path."""
    return os.path.exists(filepath)

# Example of a general utility function: Basic data info
def get_dataframe_info(df: pd.DataFrame):
    """Prints basic information about a DataFrame."""
    print("DataFrame Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nData Types:\n", df.dtypes)


# Removed redundant data loading function (handled in data_utils.py)
# Removed redundant model loading function (handled in model_utils.py)
# Removed redundant preprocessing function (handled in data_utils.py or model_utils.py specific prep)
# Removed redundant make_forecast function (handled in model_utils.py)

# Add other general utility functions as needed for the dashboard (e.g., date formatting helpers,
# simple data validation checks, configuration loading if not using constants.py for everything)

# Example: Function to get a list of unique countries from the data file without loading the whole file
# This might be useful for populating a country dropdown efficiently in the dashboard.
def get_unique_countries_from_datafile(filepath: str = DATA_PATH, country_col: str = COUNTRY_COL) -> list:
    """
    Reads the specified column from a CSV file and returns unique values.
    Useful for getting unique countries without loading the entire dataset.
    Returns an empty list if the file or column does not exist.
    """
    unique_countries = []
    if file_exists(filepath):
        try:
            # Read only the country column to save memory
            df_temp = pd.read_csv(filepath, usecols=[country_col])
            # Get unique values, convert to string to handle potential mixed types, drop NaNs, and sort
            unique_countries = sorted(df_temp[country_col].astype(str).dropna().unique().tolist())
        except Exception as e:
            print(f"Error reading unique countries from {filepath}: {e}")
    else:
        print(f"Warning: Data file not found at {filepath}")

    return unique_countries

# Example: Function to get a list of unique parameters from the data file
def get_unique_parameters_from_datafile(filepath: str = DATA_PATH, parameter_col: str = PARAMETER_COL) -> list:
     """
     Reads the specified column from a CSV file and returns unique values.
     Useful for getting unique parameters without loading the entire dataset.
     Returns an empty list if the file or column does not exist.
     """
     unique_parameters = []
     if file_exists(filepath):
         try:
             # Read only the parameter column
             df_temp = pd.read_csv(filepath, usecols=[parameter_col])
             # Get unique values, convert to string, drop NaNs, and sort
             unique_parameters = sorted(df_temp[parameter_col].astype(str).dropna().unique().tolist())
         except Exception as e:
             print(f"Error reading unique parameters from {filepath}: {e}")
     else:
         print(f"Warning: Data file not found at {filepath}")

     return unique_parameters


