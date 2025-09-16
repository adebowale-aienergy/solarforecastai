"""Plotting helpers using Plotly (suitable for Streamlit)."""

from __future__ import annotations
from typing import Optional, Dict, List
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Import constants for column names and units
from src.constants import DATE_COL, TARGET_COL, COUNTRY_COL, PARAMETER_COL, VALUE_COL, PARAMETER_UNITS


def preview_table(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top-n rows (for display)."""
    return df.head(n)


def line_actual_vs_pred(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str = "Actual vs Predicted",
    x_index: Optional[pd.Series] = None,
    y_label: str = "Value"
) -> go.Figure:
    """Generates a line plot comparing actual and predicted values."""
    if x_index is None:
        x_index = y_true.index

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_index, y=y_true, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=x_index, y=y_pred, mode="lines", name="Predicted"))
    fig.update_layout(
        title=title,
        xaxis_title="Date",  # Assuming date index
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def prophet_forecast_plot(
    forecast_df: pd.DataFrame,
    history_df: Optional[pd.DataFrame] = None,
    title: str = "Prophet Forecast",
    y_label: str = "Value",
) -> go.Figure:
    """Generates a Plotly figure for Prophet forecasts."""
    fig = go.Figure()

    # Add uncertainty band first
    if {"yhat_lower", "yhat_upper"}.issubset(forecast_df.columns):
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecast_df["ds"], forecast_df["ds"].iloc[::-1]]),
                y=pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(0,100,80,0.2)",
                name="Uncertainty",
                line=dict(width=0),
                showlegend=True,
            )
        )

    # Add forecast line
    fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat"], mode="lines", name="Forecast"))

    # Add actual history
    if history_df is not None and {"ds", "y"}.issubset(history_df.columns):
        fig.add_trace(go.Scatter(x=history_df["ds"], y=history_df["y"], mode="lines", name="Actual"))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def model_comparison_plot(series: Dict[str, pd.Series], title: str = "Model Comparison", y_label: str = "Value") -> go.Figure:
    """Generates a line plot comparing multiple time series."""
    fig = go.Figure()
    for name, y in series.items():
        fig.add_trace(go.Scatter(x=y.index, y=y, mode="lines", name=name))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def country_map(lat: float, lon: float, country_name: str, zoom: int = 3) -> go.Figure:
    """Simple country marker using scattergeo."""
    fig = go.Figure(
        go.Scattergeo(
            lon=[lon], lat=[lat], text=[country_name],
            mode="markers+text", textposition="bottom center",
            marker=dict(size=10, color="red")
        )
    )
    fig.update_geos(
        projection_type="natural earth", showcountries=True,
        lataxis_range=[-50, 75], lonaxis_range=[-180, 180]
    )
    fig.update_layout(title=f"Location: {country_name}", geo_scope="world")
    return fig


def plot_parameter_distribution_boxplot(
    df: pd.DataFrame,
    parameter: str,
    country_col: str = COUNTRY_COL,
    parameter_col: str = PARAMETER_COL,
    value_col: str = VALUE_COL
) -> go.Figure:
    """Box plot for parameter distribution across countries."""
    parameter_df = df[df[parameter_col] == parameter].copy()

    if parameter_df.empty:
        print(f"Warning: No data found for parameter '{parameter}'.")
        return go.Figure().update_layout(title=f"No data for {parameter} distribution")

    y_label = f"{parameter} ({PARAMETER_UNITS.get(parameter, 'Unknown Unit')})"

    fig = px.box(parameter_df, x=country_col, y=value_col, title=f"Distribution of {parameter} Across Countries")
    fig.update_layout(
        xaxis_title="Country",
        yaxis_title=y_label,
        xaxis={"categoryorder": "category asc"},
    )
    return fig


def plot_time_series_by_country(
    df: pd.DataFrame,
    parameter: str,
    countries: List[str],
    date_col: str = DATE_COL,
    country_col: str = COUNTRY_COL,
    parameter_col: str = PARAMETER_COL,
    value_col: str = VALUE_COL
) -> go.Figure:
    """Line plot comparing time series of a parameter for selected countries."""
    filtered_df = df[
        (df[parameter_col] == parameter) & (df[country_col].isin(countries))
    ].copy()

    if filtered_df.empty:
        print(f"Warning: No data found for parameter '{parameter}' in selected countries.")
        return go.Figure().update_layout(title=f"No data for {parameter} time series")

    filtered_df = filtered_df.sort_values(date_col)
    y_label = f"{parameter} ({PARAMETER_UNITS.get(parameter, 'Unknown Unit')})"

    fig = px.line(filtered_df, x=date_col, y=value_col, color=country_col, title=f"Time Series of {parameter}")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=y_label,
        legend_title=country_col,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
