import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set a consistent style
sns.set_style("whitegrid")

def plot_forecast(forecast_df: pd.DataFrame, actual_df: pd.DataFrame = None, title: str = "Forecast vs Actual"):
    """
    Plot forecast (from Prophet, LSTM, RF) along with actual values if provided.
    forecast_df: DataFrame with 'ds' (date) and 'yhat' (forecasted values).
    actual_df: Optional DataFrame with 'ds' (date) and 'y' (actual values).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot forecast
    if "yhat" in forecast_df.columns:
        ax.plot(forecast_df["ds"], forecast_df["yhat"], label="Forecast", color="blue")

    # Plot actuals if available
    if actual_df is not None and "y" in actual_df.columns:
        ax.plot(actual_df["ds"], actual_df["y"], label="Actual", color="orange")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Solar Power (kW)")
    ax.legend()
    return fig


def plot_actual_vs_predicted(actual, predicted, title: str = "Actual vs Predicted"):
    """
    Scatter plot comparing actual vs predicted values.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=actual, y=predicted, ax=ax, alpha=0.6)
    ax.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--', label="Perfect Fit")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend()
    return fig
