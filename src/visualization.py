# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
import plotly.express as px
from geo_data import attach_geo_data

# -------------------------------
# Matplotlib/Seaborn Plots
# -------------------------------
def plot_predictions(y_true, y_pred, title="Model Predictions"):
    """
    Line plot of actual vs predicted values
    """
    plt.figure(figsize=(8,5))
    sns.lineplot(x=range(len(y_true)), y=y_true, label="Actual", color="blue")
    sns.lineplot(x=range(len(y_pred)), y=y_pred, label="Predicted", color="red")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Solar Power")
    plt.legend()
    plt.tight_layout()
    return plt


def plot_feature_importance(model, feature_names):
    """
    Bar chart of feature importance (RandomForest, etc.)
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        feat_imp = feat_imp.sort_values("Importance", ascending=False)

        plt.figure(figsize=(8,5))
        sns.barplot(data=feat_imp, x="Importance", y="Feature", palette="viridis")
        plt.title("Feature Importance")
        plt.tight_layout()
        return plt
    else:
        raise ValueError("Model has no feature_importances_ attribute")


# -------------------------------
# Plotly (Interactive for Streamlit)
# -------------------------------
def plot_forecast_chart(df, country):
    """
    Interactive line chart of forecasts in Streamlit
    """
    st.line_chart(df.set_index("date")["forecast"], use_container_width=True)


def plot_forecast_map(df, value_col="forecast"):
    """
    Interactive geo scatter map of forecasts across countries
    """
    df_geo = attach_geo_data(df, country_col="country")
    fig = px.scatter_geo(
        df_geo,
        lat="lat",
        lon="lon",
        color=value_col,
        hover_name="country",
        size=value_col,
        projection="natural earth",
        title="Solar Forecast by Country"
    )
    st.plotly_chart(fig, use_container_width=True)
