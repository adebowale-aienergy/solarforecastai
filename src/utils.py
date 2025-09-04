import pandas as pd
import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

import os

# ===========================
# Load Dataset
# ===========================
def load_data(data_path):
    df = pd.read_csv(data_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


# ===========================
# Load ML Model
# ===========================
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"⚠️ Could not load model {model_path}: {e}")
        return None


# ===========================
# Preprocess Data
# ===========================
def preprocess_data(df):
    cols_to_keep = [
        "Date", "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN",
        "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF",
        "PSH", "T2M", "CLOUD_AMT", "AOD_55", "RH2M", "WS2M"
    ]
    df = df[[c for c in cols_to_keep if c in df.columns]]
    df = df.dropna()
    return df


# ===========================
# Forecast Helper
# ===========================
def make_forecast(model, df, horizon=7, target="ALLSKY_SFC_SW_DWN"):
    if model is None:
        # Simple baseline
        last_val = df[target].iloc[-1]
        future_dates = pd.date_range(df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=horizon)
        return pd.DataFrame({"Date": future_dates, target: [last_val] * horizon})

    # Prophet case
    if isinstance(model, Prophet):
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        return forecast[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": target})

    # LSTM case
    if hasattr(model, "predict") and "keras" in str(type(model)):
        data = df[target].values[-30:].reshape(1, 30, 1)  # use last 30 days
        preds = []
        for _ in range(horizon):
            pred = model.predict(data, verbose=0)[0][0]
            preds.append(pred)
            new_input = np.append(data[0, 1:, 0], pred)
            data = new_input.reshape(1, 30, 1)

        future_dates = pd.date_range(df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=horizon)
        return pd.DataFrame({"Date": future_dates, target: preds})

    # RandomForest or Sklearn model
    X = df.drop(columns=["Date", target], errors="ignore")
    preds = model.predict(X.tail(horizon))
    future_dates = pd.date_range(df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=horizon)
    return pd.DataFrame({"Date": future_dates, target: preds})


# ===========================
# Training Functions
# ===========================

def train_random_forest(df, target="ALLSKY_SFC_SW_DWN", save_path="rf_model.pkl"):
    X = df.drop(columns=["Date", target], errors="ignore")
    y = df[target]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    joblib.dump(model, save_path)
    print(f"✅ Random Forest model saved to {save_path}")
    return model


def train_prophet(df, target="ALLSKY_SFC_SW_DWN", save_path="prophet_model.pkl"):
    df_prophet = df.rename(columns={"Date": "ds", target: "y"})[["ds", "y"]]
    model = Prophet()
    model.fit(df_prophet)
    joblib.dump(model, save_path)
    print(f"✅ Prophet model saved to {save_path}")
    return model


def train_lstm(df, target="ALL
