# src/utils.py
import pandas as pd
import joblib
import tensorflow as tf
import pickle
import numpy as np

from prophet import Prophet

from src.constants import (
    DATA_PATH, DEFAULT_DATE_COL, DEFAULT_TARGET_COL,
    DEFAULT_COUNTRY_COL
)

# -----------------------------
# Load Data
# -----------------------------
def load_data():
    return pd.read_csv(DATA_PATH)

# -----------------------------
# Load Model
# -----------------------------
def load_model(model_path, model_type="sklearn"):
    if model_type == "sklearn":
        return joblib.load(model_path)
    elif model_type == "prophet":
        with open(model_path, "rb") as f:
            return pickle.load(f)
    elif model_type == "lstm":
        return tf.keras.models.load_model(model_path)
    else:
        raise ValueError("Unsupported model type")

# -----------------------------
# Preprocess Data
# -----------------------------
def preprocess_data(df, country, horizon):
    df = df[df[DEFAULT_COUNTRY_COL] == country].copy()
    df[DEFAULT_DATE_COL] = pd.to_datetime(df[DEFAULT_DATE_COL])
    df = df.sort_values(DEFAULT_DATE_COL)
    return df.tail(horizon)

# -----------------------------
# Make Forecast
# -----------------------------
def make_forecast(model, model_type, df, horizon):
    if model_type == "sklearn":
        X = df.drop([DEFAULT_DATE_COL, DEFAULT_TARGET_COL, DEFAULT_COUNTRY_COL], axis=1)
        preds = model.predict(X)
        return preds

    elif model_type == "prophet":
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        return forecast.tail(horizon)["yhat"].values

    elif model_type == "lstm":
        values = df[DEFAULT_TARGET_COL].values[-horizon:]
        preds = model.predict(values.reshape(1, -1, 1))
        return preds.flatten()

    else:
        raise ValueError("Unsupported model type")
