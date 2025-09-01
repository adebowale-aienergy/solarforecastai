"""Model loading and forecasting helpers.

- RF: expects scikit-learn model saved via joblib
- Prophet: expects a fitted Prophet model saved with joblib
- LSTM: expects a Keras model saved via model.save(...)
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np
import joblib

from .data_utils import split_features_target, make_prophet_frame, add_time_features
from .constants import DEFAULT_DATE_COL, DEFAULT_TARGET_COL
from sklearn.model_selection import train_test_split

# ----- Random Forest helpers -----
def load_rf_model(path: str):
    """Load scikit-learn model from joblib file."""
    return joblib.load(path)


def predict_rf(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict with scikit-learn regressor."""
    return model.predict(X)


# ----- Prophet helpers (lazy import) -----
def load_prophet_model(path: str):
    """Load Prophet model saved via joblib. Raises helpful error if Prophet not installed."""
    try:
        import prophet  # noqa: F401
    except Exception as e:
        raise RuntimeError("Prophet is not installed. Install 'prophet' to use Prophet models.") from e
    return joblib.load(path)


def forecast_prophet(model: Any, periods: int = 30, freq: str = "D"):
    """Create future dataframe and forecast using a fitted Prophet model."""
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast


# ----- LSTM helpers (lazy import) -----
def load_lstm_model(path: str):
    """Load a Keras/TensorFlow model saved with model.save()."""
    try:
        import tensorflow as _tf  # noqa: F401
        from tensorflow.keras.models import load_model as _load_model
    except Exception as e:
        raise RuntimeError("TensorFlow/Keras is not installed. Install 'tensorflow' to use LSTM models.") from e
    return _load_model(path)


def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Create 3D sequences for LSTM training/prediction."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def predict_lstm(model: Any, X_seq: np.ndarray) -> np.ndarray:
    """Predict with a Keras LSTM model and return flattened array."""
    out = model.predict(X_seq, verbose=0)
    return out.flatten()


# ----- Unified forecasting -----
def make_forecast(
    model: Any,
    model_name: str,
    df,
    horizon: int = DEFAULT_DATE_COL,  # will usually be replaced
    date_col: str = DEFAULT_DATE_COL,
    target_col: str = DEFAULT_TARGET_COL,
    drop_cols: Optional[list] = None,
    lstm_time_steps: int = 10,
) -> Dict[str, Any]:
    """
    Unified interface for producing forecasts/predictions.
    Returns a dict containing:
      - 'forecast_df' : (Prophet) DataFrame with forecast (ds,yhat,...)
      - 'y_true' : array-like of true values used for evaluation (if applicable)
      - 'y_pred' : array-like of predictions aligned to y_true (if applicable)
    Notes:
      - For RF and LSTM we do a non-shuffle train/test split (last 20% used as test).
      - RF predictions are single-step on test-set (not recursive multi-step).
      - LSTM predictions here use sequence creation (time_steps) and evaluate on last portion.
      - Prophet returns future forecast including the horizon.
    """
    out = {"forecast_df": None, "y_true": None, "y_pred": None}

    # Prophet path
    if model_name.lower() in ("prophet",):
        prophet_df = make_prophet_frame(df, date_col=date_col, target_col=target_col)
        forecast = forecast_prophet(model, periods=horizon)
        out["forecast_df"] = forecast
        return out

    # RF / LSTM path
    df_proc = df.copy()
    if date_col in df_proc.columns:
        df_proc = add_time_features(df_proc, date_col=date_col, include_cyclical=True)

    X_df, y_ser = split_features_target(df_proc, target_col=target_col, drop_cols=[date_col] if date_col in df_proc.columns else None)
    X_arr = X_df.values
    y_arr = y_ser.values

    if len(X_arr) < 10:
        raise ValueError("Not enough rows to produce train/test split.")

    # non-shuffled split so time order preserved
    split_idx = int(len(X_arr) * 0.8)
    X_train, X_test = X_arr[:split_idx], X_arr[split_idx:]
    y_train, y_test = y_arr[:split_idx], y_arr[split_idx:]

    if model_name.lower() in ("random forest", "rf"):
        y_pred = predict_rf(model, X_test)
        out["y_true"] = y_test
        out["y_pred"] = y_pred
        return out

    if model_name.lower() in ("lstm",):
        # Create sequences using entire array (so sequences align to original order)
        Xs, ys = create_sequences(X_arr, y_arr, time_steps=lstm_time_steps)
        if len(Xs) == 0:
            raise ValueError("Not enough rows to create LSTM sequences with the requested time_steps.")
        split_idx_seq = int(len(Xs) * 0.8)
        Xs_test = Xs[split_idx_seq:]
        ys_test = ys[split_idx_seq:]
        y_pred = predict_lstm(model, Xs_test)
        out["y_true"] = ys_test
        out["y_pred"] = y_pred
        return out

    raise ValueError(f"Unknown model_name: {model_name}")
