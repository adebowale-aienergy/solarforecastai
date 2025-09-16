"""Model building, training, saving, loading, and forecasting utilities."""

# Standard library imports
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import joblib
import json  # Added for Prophet JSON handling

# Third-party imports
from sklearn.ensemble import RandomForestRegressor
# Prophet and TensorFlow/Keras will be lazily imported in their respective functions

# Local imports
from src.constants import (
    RF_MODEL_PATH, PROPHET_MODEL_PATH, LSTM_MODEL_PATH,
    DATE_COL, TARGET_COL, VALUE_COL,
    RF_FEATURES, PROPHET_COLS, LSTM_FEATURES
)

# ----- Model Training Functions -----

def train_random_forest_model(X: pd.DataFrame, y: pd.Series, **kwargs) -> RandomForestRegressor:
    """Trains a Random Forest Regressor model."""
    model = RandomForestRegressor(**kwargs)
    model.fit(X, y)
    return model

def train_prophet_model(df_prophet: pd.DataFrame, **kwargs) -> Any:
    """Trains a Prophet model."""
    try:
        from prophet import Prophet
    except ImportError:
        raise RuntimeError("Prophet is not installed. Install 'prophet' to train Prophet models.")

    if 'ds' not in df_prophet.columns or 'y' not in df_prophet.columns:
        raise ValueError("df_prophet must contain 'ds' and 'y' columns.")
    if not pd.api.types.is_numeric_dtype(df_prophet['y']):
        raise ValueError("'y' column in df_prophet must be numeric.")

    model = Prophet(**kwargs)
    model.fit(df_prophet)
    return model

def train_lstm_model(X_seq: np.ndarray, y_seq: np.ndarray, epochs: int = 50, batch_size: int = 32,
                     validation_split: float = 0.2, **kwargs) -> Any:
    """Trains an LSTM model."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
    except ImportError:
        raise RuntimeError("TensorFlow/Keras is not installed. Install 'tensorflow' to train LSTM models.")

    if X_seq.ndim != 3 or y_seq.ndim != 1 or X_seq.shape[0] != y_seq.shape[0]:
        raise ValueError(f"Invalid input shapes for LSTM training. "
                         f"X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer=kwargs.get('optimizer', 'adam'),
                  loss=kwargs.get('loss', 'mse'))
    model.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size,
              validation_split=validation_split, verbose=0)

    return model

# ----- Model Saving Functions -----

def save_random_forest_model(model: RandomForestRegressor, path: str = RF_MODEL_PATH):
    """Save scikit-learn model to a joblib file."""
    joblib.dump(model, path)

def save_prophet_model(model: Any, path: str = PROPHET_MODEL_PATH):
    """Save Prophet model to a JSON file using Prophet's built-in serialization."""
    try:
        from prophet.serialize import model_to_json
    except ImportError:
        raise RuntimeError("Prophet is not installed. Cannot save Prophet model.")

    with open(path, 'w') as fout:
        fout.write(model_to_json(model))

def save_lstm_model(model: Any, path: str = LSTM_MODEL_PATH):
    """Save a Keras/TensorFlow model using model.save()."""
    try:
        from tensorflow.keras.models import Model as KerasModel
        if not isinstance(model, KerasModel):
            print("Warning: Provided model does not seem to be a Keras Model instance.")
        model.save(path)
    except ImportError:
        raise RuntimeError("TensorFlow/Keras is not installed. Cannot save LSTM model.")
    except Exception as e:
        print(f"Error saving LSTM model: {e}")

# ----- Model Loading Functions -----

def load_random_forest_model(path: str = RF_MODEL_PATH) -> RandomForestRegressor:
    """Load scikit-learn model from joblib file."""
    return joblib.load(path)

def load_prophet_model(path: str = PROPHET_MODEL_PATH) -> Any:
    """Load Prophet model from a JSON file."""
    try:
        from prophet.serialize import model_from_json
    except ImportError:
        raise RuntimeError("Prophet is not installed. Cannot load Prophet model.")
    with open(path, 'r') as fin:
        return model_from_json(fin.read())

def load_lstm_model(path: str = LSTM_MODEL_PATH) -> Any:
    """Load a Keras/TensorFlow model saved with model.save()."""
    try:
        from tensorflow.keras.models import load_model as _load_model
    except ImportError:
        raise RuntimeError("TensorFlow/Keras is not installed. Cannot load LSTM model.")
    return _load_model(path)

# ----- Prediction/Forecasting Functions -----

def predict_random_forest(model: RandomForestRegressor, X: pd.DataFrame) -> np.ndarray:
    """Predict with scikit-learn regressor."""
    return model.predict(X)

def forecast_prophet(model: Any, periods: int, freq: str = "D") -> pd.DataFrame:
    """Forecast using a fitted Prophet model."""
    try:
        from prophet import Prophet as ProphetModel
        if not isinstance(model, ProphetModel):
            print("Warning: Provided model is not a Prophet instance.")
    except ImportError:
        pass

    future = model.make_future_dataframe(periods=periods, freq=freq)
    return model.predict(future)

def predict_lstm(model: Any, X_seq: np.ndarray) -> np.ndarray:
    """Predict with a Keras LSTM model and return flattened array."""
    if not hasattr(model, 'predict'):
        raise ValueError("Provided model does not have a 'predict' method.")
    if X_seq.ndim != 3:
        raise ValueError(f"Invalid input shape for LSTM prediction. Expected 3D, got {X_seq.ndim}D.")
    return model.predict(X_seq, verbose=0).flatten()

# ----- Helper for LSTM sequence creation -----

def create_sequences(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series,
                     time_steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Create 3D sequences for LSTM training/prediction."""
    X, y = np.asarray(X), np.asarray(y)
    Xs, ys = [], []

    if len(X) <= time_steps:
        feature_dim = X.shape[-1] if X.ndim > 1 and X.shape[0] > 0 else 1
        return np.array([]).reshape(0, time_steps, feature_dim), np.array([])

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    feature_dim = X.shape[1]
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps), :])
        ys.append(y[i + time_steps])

    if not Xs:
        return np.array([]).reshape(0, time_steps, feature_dim), np.array([])

    return np.array(Xs), np.array(ys)
