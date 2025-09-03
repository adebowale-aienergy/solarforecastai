# src/eval_utils.py
import matplotlib.pyplot as plt
import io
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def plot_evaluation_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAE:** {mae:.2f}")

    fig, ax = plt.subplots()
    ax.plot(y_true, label="Actual")
    ax.plot(y_pred, label="Predicted")
    ax.legend()
    st.pyplot(fig)
