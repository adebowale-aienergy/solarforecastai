"""Evaluation utilities for forecasting models."""

from __future__ import annotations
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd # Import pandas for potential Series/DataFrame handling
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Import r2_score
# import matplotlib.pyplot as plt # Keep if planning to generate plots within functions
# import io # Keep if needed for handling plot output
# import streamlit as st # Keep if functions are designed for Streamlit display

def calculate_regression_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict:
    """
    Calculates standard regression evaluation metrics.

    Args:
        y_true: Array or Series of true target values.
        y_pred: Array or Series of predicted target values.

    Returns:
        A dictionary containing the calculated metrics (MAE, MSE, RMSE, R-squared).
        Returns NaN for metrics if inputs are invalid or empty.
    """
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        print("Warning: Invalid input arrays for metrics calculation.")
        return {
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "R-squared": np.nan
        }

    try:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred) # Calculate R-squared

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R-squared": r2
        }
    except Exception as e:
        print(f"An error occurred during metrics calculation: {e}")
        return {
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "R-squared": np.nan
        }


# The original plot_evaluation_metrics function seems designed for direct Streamlit use.
# It's often better to separate calculation from presentation.
# If needed, a plotting function that accepts metrics and/or y_true/y_pred can be added here,
# or handle plotting directly in the Streamlit app using the calculated metrics.

# Keeping the original plotting function as an example, but it might need adjustments
# depending on where it's called (e.g., if not directly in a Streamlit context).
# Assuming for dashboard use, it will be called within the Streamlit app.
# If you need a more general plotting function not tied to Streamlit, create a new one.
# def plot_evaluation_metrics(y_true, y_pred):
#     """
#     Plots actual vs predicted values and displays key metrics in Streamlit.
#     Assumes this function is called within a Streamlit application.
#     """
#     # It might be better to call calculate_regression_metrics here first
#     metrics = calculate_regression_metrics(y_true, y_pred)

#     st.write(f"**RMSE:** {metrics.get('RMSE', np.nan):.2f}")
#     st.write(f"**MAE:** {metrics.get('MAE', np.nan):.2f}")
#     st.write(f"**R-squared:** {metrics.get('R-squared', np.nan):.2f}") # Display R-squared

#     fig, ax = plt.subplots(figsize=(12, 6)) # Adjust figure size
#     ax.plot(y_true.index if isinstance(y_true, pd.Series) else range(len(y_true)), y_true, label="Actual")
#     ax.plot(y_pred.index if isinstance(y_pred, pd.Series) else range(len(y_pred)), y_pred, label="Predicted")
#     ax.set_title("Actual vs Predicted Values")
#     ax.set_xlabel("Time or Index")
#     ax.set_ylabel("Value") # Use a generic label as parameter varies
#     ax.legend()
#     ax.grid(True)
#     st.pyplot(fig)


