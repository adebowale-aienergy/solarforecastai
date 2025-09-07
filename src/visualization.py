import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_forecast(dates, actual, forecast, model_name):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(dates, actual, label="Actual", color="blue")
    ax.plot(dates, forecast, label="Forecast", color="orange")
    ax.set_title(f"{model_name} Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Solar Radiation")
    ax.legend()
    st.pyplot(fig)

def plot_distribution(df, col):
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")
    st.pyplot(fig)
