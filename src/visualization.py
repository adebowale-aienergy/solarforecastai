"""Plotting helpers using Plotly (suitable for Streamlit)."""

from __future__ import annotations
from typing import Optional, Dict, Any, Sequence, List
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Import constants for column names and units
from constants import DATE_COL, TARGET_COL, COUNTRY_COL, PARAMETER_COL, VALUE_COL, PARAMETER_UNITS


def preview_table(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top-n rows (for display)."""
    return df.head(n)


def line_actual_vs_pred(y_true: pd.Series, y_pred: pd.Series, title: str = "Actual vs Predicted", x_index: Optional[pd.Series] = None, y_label: str = "Value") -> go.Figure:
    """
    Generates a line plot comparing actual and predicted values.

    Args:
        y_true: Pandas Series of true target values with date index.
        y_pred: Pandas Series of predicted target values with date index.
        title: Title of the plot.
        x_index: Optional Pandas Series for the x-axis (e.g., dates). Defaults to index of y_true.
        y_label: Label for the y-axis.

    Returns:
        A Plotly Figure object.
    """
    if x_index is None:
        x_index = y_true.index

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_index, y=y_true, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=x_index, y=y_pred, mode="lines", name="Predicted"))
    fig.update_layout(
        title=title,
        xaxis_title="Date", # Assuming date index
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # Position legend at the top
    )
    return fig


def prophet_forecast_plot(forecast_df: pd.DataFrame, history_df: Optional[pd.DataFrame] = None, title: str = "Prophet Forecast", y_label: str = "Value") -> go.Figure:
    """
    Generates a Plotly figure for Prophet forecasts.

    Args:
        forecast_df: DataFrame with 'ds', 'yhat', 'yhat_lower', 'yhat_upper' columns.
        history_df: Optional DataFrame with 'ds' and 'y' (actual) to plot alongside.
        title: Title of the plot.
        y_label: Label for the y-axis.

    Returns:
        A Plotly Figure object.
    """
    fig = go.Figure()

    # Add uncertainty band first so it's in the background
    if {"yhat_lower", "yhat_upper"}.issubset(forecast_df.columns):
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecast_df["ds"], forecast_df["ds"].iloc[::-1]]), # Ensure dates are ordered for fill
                y=pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"].iloc[::-1]]),
                fill="toself",
                fillcolor='rgba(0,100,80,0.2)', # Greenish fill with transparency
                name="Uncertainty",
                line=dict(width=0),
                showlegend=True,
            )
        )
    # Add forecast line
    fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat"], mode="lines", name="Forecast"))

    # Add actual history data
    if history_df is not None and {"ds", "y"}.issubset(history_df.columns):
        fig.add_trace(go.Scatter(x=history_df["ds"], y=history_df["y"], mode="lines", name="Actual"))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def model_comparison_plot(series: Dict[str, pd.Series], title: str = "Model Comparison", y_label: str = "Value") -> go.Figure:
    """
    Generates a line plot comparing multiple time series (e.g., actual vs different model forecasts).

    Args:
        series: Dictionary where keys are series names (e.g., 'Actual', 'RF Forecast')
                and values are Pandas Series with a date index.
        title: Title of the plot.
        y_label: Label for the y-axis.

    Returns:
        A Plotly Figure object.
    """
    fig = go.Figure()
    for name, y in series.items():
        fig.add_trace(go.Scatter(x=y.index, y=y, mode="lines", name=name))

    fig.update_layout(
        title=title,
        xaxis_title="Date", # Assuming date index
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def country_map(lat: float, lon: float, country_name: str, zoom: int = 3) -> go.Figure:
    """Simple country marker using scattergeo (no token required)."""
    fig = go.Figure(go.Scattergeo(lon=[lon], lat=[lat], text=[country_name], mode="markers+text", textposition="bottom center", marker=dict(size=10, color="red")))
    fig.update_geos(projection_type="natural earth", showcountries=True, lataxis_range=[-50, 75], lonaxis_range=[-180, 180]) # Adjust ranges for better global view
    fig.update_layout(title=f"Location: {country_name}", geo_scope='world') # Use 'world' scope
    return fig

def plot_parameter_distribution_boxplot(df: pd.DataFrame, parameter: str, country_col: str = COUNTRY_COL, parameter_col: str = PARAMETER_COL, value_col: str = VALUE_COL) -> go.Figure:
    """
    Generates a box plot to show the distribution of a specific parameter across countries.

    Args:
        df: DataFrame in long format with country, parameter, and value columns.
        parameter: The specific climate parameter to plot.
        country_col: Name of the column containing country names.
        parameter_col: Name of the column containing parameter names.
        value_col: Name of the column containing parameter values.

    Returns:
        A Plotly Figure object.
    """
    parameter_df = df[df[parameter_col] == parameter].copy()

    if parameter_df.empty:
        print(f"Warning: No data found for parameter '{parameter}' for plotting distribution.")
        return go.Figure().update_layout(title=f"No data for {parameter} distribution")

    y_label = f"{parameter} ({PARAMETER_UNITS.get(parameter, 'Unknown Unit')})"

    fig = px.box(parameter_df, x=country_col, y=value_col, title=f"Distribution of {parameter} Across Countries")
    fig.update_layout(
        xaxis_title="Country",
        yaxis_title=y_label,
        xaxis={'categoryorder':'category asc'} # Ensure countries are sorted alphabetically
    )
    return fig

def plot_time_series_by_country(df: pd.DataFrame, parameter: str, countries: List[str], date_col: str = DATE_COL, country_col: str = COUNTRY_COL, parameter_col: str = PARAMETER_COL, value_col: str = VALUE_COL) -> go.Figure:
    """
    Generates a line plot comparing the time series of a specific parameter for selected countries.

    Args:
        df: DataFrame in long format with date, country, parameter, and value columns.
        parameter: The specific climate parameter to plot.
        countries: List of country names to include in the plot.
        date_col: Name of the column containing dates.
        country_col: Name of the column containing country names.
        parameter_col: Name of the column containing parameter names.
        value_col: Name of the column containing parameter values.

    Returns:
        A Plotly Figure object.
    """
    filtered_df = df[
        (df[parameter_col] == parameter) &
        (df[country_col].isin(countries))
    ].copy()

    if filtered_df.empty:
        print(f"Warning: No data found for parameter '{parameter}' and selected countries for time series plot.")
        return go.Figure().update_layout(title=f"No data for {parameter} time series in selected countries")


    # Sort by date for proper time series plotting
    filtered_df = filtered_df.sort_values(date_col)

    y_label = f"{parameter} ({PARAMETER_UNITS.get(parameter, 'Unknown Unit')})"


    fig = px.line(filtered_df, x=date_col, y=value_col, color=country_col, title=f"Time Series of {parameter} in Selected Countries")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=y_label,
        legend_title=country_col,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

