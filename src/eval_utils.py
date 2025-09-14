import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def plot_evaluation_metrics(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    st.write(f"### {model_name} Evaluation")
    st.write(f"- MAE: {mae:.2f}")
    st.write(f"- RMSE: {rmse:.2f}")

    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.set_title(f"{model_name} Predictions vs Actual")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)
