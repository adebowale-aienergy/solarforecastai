from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# --------- Random Forest ----------
def load_rf_model(path: str):
    """Load a scikit-learn RandomForest (or compatible) model."""
    return joblib.load(path)

def predict_rf(model, X) -> np.ndarray:
    """Predict with Random Forest."""
    return model.predict(X)

# --------- Prophet (lazy import) ----------
def load_prophet_model(path: str):
    """
    Load a serialized Prophet model saved via joblib.
    Prophet must be installed in environment.
    """
    try:
        from prophet import Prophet  # noqa: F401
    except Exception as e:
        raise RuntimeError("Prophet is not installed. Please add to requirements.") from e
    return joblib.load(path)

def forecast_prophet(model, periods: int = 30, freq: str = "D"):
    """
    Make future dataframe using fitted Prophet model and return forecast df.
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

# --------- LSTM / Keras (lazy import) ----------
def load_lstm_model(path: str):
    """
    Load a Keras/TensorFlow model saved with model.save(...).
    TensorFlow must be installed in environment.
    """
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras.models import load_model
    except Exception as e:
        raise RuntimeError("TensorFlow/Keras is not installed. Add to requirements.") from e
    return load_model(path)

def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Turn tabular X and target y into sequences for LSTM.
    Returns X_seq: (n_samples - time_steps, time_steps, n_features), y_seq.
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def predict_lstm(model, X_seq: np.ndarray) -> np.ndarray:
    """Predict with LSTM model on sequence data."""
    return model.predict(X_seq)

# --------- Scalers ----------
def fit_feature_scaler(X: np.ndarray, kind: str = "minmax") -> object:
    """
    Fit and return a scaler (minmax or standard).
    """
    if kind == "minmax":
        scaler = MinMaxScaler()
    elif kind == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("Scaler kind must be 'minmax' or 'standard'.")
    return scaler.fit(X)

def transform_with_scaler(scaler, X: np.ndarray) -> np.ndarray:
    """Transform features with a fitted scaler."""
    return scaler.transform(X)
