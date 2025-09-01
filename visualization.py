"""Plotting helpers using Plotly (suitable for Streamlit)."""

from __future__ import annotations
from typing import Optional, Dict, Any, Sequence
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def preview_table(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top-n rows (for display)."""
    return df.head(n)


def line_actual_vs_pred(y_true: Sequence, y_pred: Sequence, title: str = "Actual vs Predicted", x_index: Optional[Sequence] = None) -> go.Figure:
    if x_index is None:
        x_index = list(range(len(y_true)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_index, y=y_true, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=x_index, y=y_pred, mode="lines", name="Predicted"))
    fig.update_layout(title=title, xaxis_title="Index/Time", yaxis_title="Value", legend=dict(orientation="h"))
    return fig


def prophet_forecast_plot(forecast_df: pd.DataFrame, history_df: Optional[pd.DataFrame] = None, title: str = "Prophet Forecast") -> go.Figure:
    """
    forecast_df expected to include columns: ds, yhat, optionally yhat_lower, yhat_upper.
    history_df (optional): dataframe with ds and y (actual) to plot alongside.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat"], mode="lines", name="yhat"))
    if {"yhat_lower", "yhat_upper"}.issubset(forecast_df.columns):
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecast_df["ds"], forecast_df["ds"][::-1]]),
                y=pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"][::-1]]),
                fill="toself",
                name="Uncertainty",
                opacity=0.2,
                line=dict(width=0),
                showlegend=True,
            )
        )
    if history_df is not None and {"ds", "y"}.issubset(history_df.columns):
        fig.add_trace(go.Scatter(x=history_df["ds"], y=history_df["y"], mode="lines", name="Actual"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Value", legend=dict(orientation="h"))
    return fig


def model_comparison_plot(series: Dict[str, Sequence], title: str = "Model Comparison", x_index: Optional[Sequence] = None) -> go.Figure:
    fig = go.Figure()
    if x_index is None:
        first = next(iter(series.values()))
        x_index = list(range(len(first)))
    for name, y in series.items():
        fig.add_trace(go.Scatter(x=x_index, y=y, mode="lines", name=name))
    fig.update_layout(title=title, xaxis_title="Index/Time", yaxis_title="Value", legend=dict(orientation="h"))
    return fig


def country_map(lat: float, lon: float, country_name: str, zoom: int = 2) -> go.Figure:
    """Simple country marker using scattergeo (no token required)."""
    fig = go.Figure(go.Scattergeo(lon=[lon], lat=[lat], text=[country_name], mode="markers+text", textposition="bottom center"))
    fig.update_geos(projection_type="natural earth", showcountries=True)
    fig.update_layout(title=f"Location: {country_name}")
    return fig
