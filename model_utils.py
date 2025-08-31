from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
import numpy as np
import joblib

from .data_utils import split_features_target, make_prophet_frame, add_time_features
from .constants import DEFAULT_DATE_COL, DEFAULT_TARGET_COL
from sklearn.model_selection import train_test_split

# ----- RF helpers -----
def load_rf_model(path: str):
    """Load a scikit-learn model saved via joblib."""
    return joblib.load(path)

def predict_rf(model, X: np.ndarray) -> np.ndarray:
    """Predict with RF (or scikit-learn regressor)."""
    return model.predict(X)

# ----- Prophet helpers -----
def load_prophet_model(path: str):
    """Load a Prophet model serialized with joblib. Requires prophet installed."""
    try:
        import prophet  # noqa: F401
    except Exception as e:
        raise RuntimeError("Prophet not installed. Add 'prophet' to requirements.") from e
    return joblib.load(path)

def forecast_prophet(model, df_prophet, periods: int = 30, freq: str = "D"):
    """Return Prophet forecast DataFrame (model must be already fitted)."""
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

# ----- LSTM helpers -----
def load_lstm_model(path: str):
    """Load a Keras/TensorFlow model saved via model.save(...)."""
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras.models import load_model
    except Exception as e:
        raise RuntimeError("TensorFlow not installed. Add 'tensorflow' to requirements.") from e
    return load_model(path)

def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM training/prediction."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def predict_lstm(model, X_seq: np.ndarray) -> np.ndarray:
    """Predict with loaded LSTM model."""
    return model.predict(X_seq).flatten()

# ----- Convenience: make_forecast -----
def make_forecast(model: Any, model_name: str, df, horizon: int = 30, date_col: str = DEFAULT_DATE_COL, target_col: str = DEFAULT_TARGET_COL, drop_cols: Optional[list] = None) -> Dict[str, Any]:
    """
    Unified forecast interface.
    Returns dict with keys:
      - 'forecast_df' (for Prophet) or None
      - 'y_true' (series used for evaluation - test set)
      - 'y_pred' (predictions aligned to y_true)
    Notes:
      - RF and LSTM return predictions on a holdout test split (80/20),
        not recursive multi-step forecasts. Implement recursive logic if you need future multi-step predictions.
      - Prophet returns forecast dataframe including future horizon.
    """
    res = {"forecast_df": None, "y_true": None, "y_pred": None}
    if model_name.lower() == "prophet":
        # Expect df contains date_col and target_col
        prophet_df = make_prophet_frame(df, date_col=date_col, target_col=target_col)
        forecast = forecast_prophet(model, prophet_df, periods=horizon)
        res["forecast_df"] = forecast
        # If model has history, extract overlap to compute simple eval (last len(history) vs forecast start)
        return res

    # For RF and LSTM: build features, split, predict on test split
    df_proc = add_time_features(df, date_col=date_col, include_cyclical=True)
    X, y = split_features_target(df_proc, target_col=target_col, drop_cols=[date_col])
    if len(X) < 10:
        raise ValueError("Not enough rows to produce train/test split for RF/LSTM predictions.")

    X_arr = X.values
    y_arr = y.values

    X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=0.2, shuffle=False)
    if model_name.lower() == "random forest" or model_name.lower() == "rf":
        y_pred = predict_rf(model, X_test)
        res["y_true"] = y_test
        res["y_pred"] = y_pred
        return res

    if model_name.lower() == "lstm":
        # Create sequences from full array with default time_steps=10
        time_steps = 10
        Xs, ys = create_sequences(X_arr, y_arr, time_steps=time_steps)
        split_idx = int(len(Xs) * 0.8)
        Xs_test = Xs[split_idx:]
        ys_test = ys[split_idx:]
        if Xs_test.size == 0:
            raise ValueError("Not enough sequence rows for LSTM test split.")
        y_pred = predict_lstm(model, Xs_test)
        res["y_true"] = ys_test
        res["y_pred"] = y_pred
        return res

    raise ValueError("Unknown model_name provided to make_forecast.")
