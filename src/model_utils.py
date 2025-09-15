"""Model building, training, saving, loading, and forecasting utilities."""

# Standard library imports
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import joblib
import json # Added for Prophet JSON handling

# Third-party imports
from sklearn.ensemble import RandomForestRegressor
# Prophet and TensorFlow/Keras will be lazily imported in their respective functions

# Local imports
from constants import (
    RF_MODEL_PATH, PROPHET_MODEL_PATH, LSTM_MODEL_PATH,
    DATE_COL, TARGET_COL, VALUE_COL,
    RF_FEATURES, PROPHET_COLS, LSTM_FEATURES
)

# ----- Model Training Functions -----

def train_random_forest_model(X: pd.DataFrame, y: pd.Series, **kwargs) -> RandomForestRegressor:
    """
    Trains a Random Forest Regressor model.

    Args:
        X: DataFrame of features.
        y: Series of target values.
        **kwargs: Additional arguments for RandomForestRegressor.

    Returns:
        A trained RandomForestRegressor model.
    """
    model = RandomForestRegressor(**kwargs)
    model.fit(X, y)
    return model

def train_prophet_model(df_prophet: pd.DataFrame, **kwargs) -> Any:
    """
    Trains a Prophet model.

    Args:
        df_prophet: DataFrame with 'ds' (datetime) and 'y' (numeric) columns.
        **kwargs: Additional arguments for Prophet.

    Returns:
        A trained Prophet model.
    Raises:
        RuntimeError: If Prophet is not installed.
        ValueError: If input columns have incorrect types or are missing.
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise RuntimeError("Prophet is not installed. Install 'prophet' to train Prophet models.")

    # Prophet requires 'ds' and 'y' columns
    if 'ds' not in df_prophet.columns or 'y' not in df_prophet.columns:
        raise ValueError("df_prophet must contain 'ds' and 'y' columns.")
    if not pd.api.types.is_numeric_dtype(df_prophet['y']):
         raise ValueError("'y' column in df_prophet must be numeric.")
    # Allow object type for 'ds' initially, as pd.to_datetime handles various formats,
    # but ensure it can be converted. A more robust check could be added if needed.
    # if not pd.api.types.is_datetime64_any_dtype(df_prophet['ds']):
    #      raise ValueError("'ds' column in df_prophet must be datetime.")

    model = Prophet(**kwargs)
    model.fit(df_prophet)
    return model

def train_lstm_model(X_seq: np.ndarray, y_seq: np.ndarray, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2, **kwargs) -> Any:
    """
    Trains an LSTM model.

    Args:
        X_seq: 3D numpy array of input sequences (samples, time_steps, features).
        y_seq: 1D numpy array of target values.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        validation_split: Fraction of data to use for validation.
        **kwargs: Additional arguments for model compilation or fitting.

    Returns:
        A trained Keras LSTM model.
    Raises:
        RuntimeError: If TensorFlow/Keras is not installed.
        ValueError: If input shapes are incorrect.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
    except ImportError:
        raise RuntimeError("TensorFlow/Keras is not installed. Install 'tensorflow' to train LSTM models.")

    if X_seq.ndim != 3 or y_seq.ndim != 1 or X_seq.shape[0] != y_seq.shape[0]:
         raise ValueError(f"Invalid input shapes for LSTM training. X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")


    model = Sequential()
    # Example LSTM architecture - can be modified
    # Use the shape of the input sequences for input_shape
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) # Output layer for regression

    # Compile the model - allow overriding optimizer/loss via kwargs
    model.compile(optimizer=kwargs.get('optimizer', 'adam'), loss=kwargs.get('loss', 'mse'))

    # Train the model
    # Using verbose=0 to avoid printing training progress during execution
    model.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)

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
        # Check if it's a Keras model instance before saving
        from tensorflow.keras.models import Model as KerasModel
        if not isinstance(model, KerasModel):
             print("Warning: Provided model does not seem to be a Keras Model instance.")
             # Attempt to save anyway, but warn the user
        model.save(path)
    except ImportError:
        raise RuntimeError("TensorFlow/Keras is not installed. Cannot save LSTM model.")
    except Exception as e:
        print(f"Error saving LSTM model: {e}")
        # Depending on requirements, you might want to re-raise the exception
        # raise


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
    try:
        with open(path, 'r') as fin:
            model = model_from_json(fin.read())
        return model
    except FileNotFoundError:
        print(f"Error: Prophet model file not found at {path}")
        raise
    except Exception as e:
        print(f"Error loading Prophet model from {path}: {e}")
        raise # Re-raise the exception

def load_lstm_model(path: str = LSTM_MODEL_PATH) -> Any:
    """Load a Keras/TensorFlow model saved with model.save()."""
    try:
        from tensorflow.keras.models import load_model as _load_model
    except ImportError:
        raise RuntimeError("TensorFlow/Keras is not installed. Cannot load LSTM model.")
    try:
        return _load_model(path)
    except FileNotFoundError:
        print(f"Error: LSTM model file not found at {path}")
        raise
    except Exception as e:
        print(f"Error loading LSTM model from {path}: {e}")
        raise # Re-raise the exception


# ----- Prediction/Forecasting Functions -----

def predict_random_forest(model: RandomForestRegressor, X: pd.DataFrame) -> np.ndarray:
    """Predict with scikit-learn regressor."""
    return model.predict(X)

def forecast_prophet(model: Any, periods: int, freq: str = "D") -> pd.DataFrame:
    """Create future dataframe and forecast using a fitted Prophet model."""
    try:
        # Check if the model is a Prophet model instance (basic check)
        from prophet import Prophet as ProphetModel
        if not isinstance(model, ProphetModel):
             print("Warning: Provided model does not seem to be a Prophet instance for forecasting.")
             # Attempt to forecast anyway if it has the necessary methods
    except ImportError:
        # If prophet isn't installed, we can't even do the instance check,
        # but the load/train functions should have already raised an error.
        pass # Continue assuming the model object might work or will fail gracefully

    # Prophet's make_future_dataframe uses the last date in the training data
    # and extends from there. It doesn't need the original dataframe here,
    # only the number of periods and frequency.
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

def predict_lstm(model: Any, X_seq: np.ndarray) -> np.ndarray:
    """Predict with a Keras LSTM model and return flattened array."""
    try:
        # Check if the model has a predict method (basic check for Keras models)
        if not hasattr(model, 'predict'):
            raise ValueError("Provided model does not have a 'predict' method.")
        if X_seq.ndim != 3:
             raise ValueError(f"Invalid input shape for LSTM prediction. Expected 3D, got {X_seq.ndim}D.")

        out = model.predict(X_seq, verbose=0)
        # The output shape of a regression LSTM is typically (samples, 1)
        # Flatten to a 1D array for consistency with other model outputs
        return out.flatten()
    except ImportError:
         raise RuntimeError("TensorFlow/Keras is not installed. Cannot perform LSTM prediction.")
    except Exception as e:
        print(f"Error during LSTM prediction: {e}")
        raise # Re-raise the exception


# ----- Helper for LSTM sequence creation -----
def create_sequences(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, time_steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 3D sequences for LSTM training/prediction.

    Args:
        X: Input features. Can be a numpy array or pandas DataFrame.
        y: Target values. Can be a numpy array or pandas Series.
        time_steps: The number of time steps in each sequence.

    Returns:
        A tuple containing:
            - Xs: 3D numpy array of input sequences (samples, time_steps, features).
            - ys: 1D numpy array of target values corresponding to the end of each sequence.
        Returns empty arrays if not enough data to create sequences.
    """
    Xs, ys = [], []
    # Ensure X and y are numpy arrays for consistent indexing
    X = np.asarray(X)
    y = np.asarray(y)

    # LSTM sequence creation requires at least time_steps data points to create one sequence of time_steps length.
    # To predict the next step (y), we need time_steps + 1 data points in total.
    if len(X) <= time_steps:
        print(f"Warning: Not enough data ({len(X)}) to create sequences with time_steps={time_steps}. Need at least {time_steps + 1} data points for training data. Returning empty arrays.")
        # Adjust shape to match expected 3D output shape even if empty
        # If X was originally empty or 1D, X.shape could be (0,) or (n,). Handle this.
        feature_dim = X.shape[-1] if X.ndim > 1 and X.shape[0] > 0 else (X.shape[0] if X.ndim == 1 and X.shape[0] > 0 else 1)
        # If X is completely empty, feature_dim should probably be inferred from expected input features,
        # but without context of expected features here, default to 1 or handle based on X.shape if possible.
        # If X is truly empty from the start, X.shape is (0,) or (0, num_features).
        if X.shape[0] == 0:
             feature_dim = 1 # Default feature dimension if input X is empty

        return np.array([]).reshape(0, time_steps, feature_dim), np.array([])


    # Handle case where X is 1D (e.g., only the value column)
    if X.ndim == 1:
        X = X.reshape(-1, 1) # Reshape to (samples, 1)

    # Determine feature dimension after potential reshaping
    feature_dim = X.shape[1]


    for i in range(len(X) - time_steps):
        # Ensure the slice is taken correctly for multi-dimensional X
        Xs.append(X[i:(i + time_steps), :])
        ys.append(y[i + time_steps])

    # If Xs is still empty after the loop (e.g., len(X) == time_steps), return empty arrays with correct shape
    if not Xs:
         return np.array([]).reshape(0, time_steps, feature_dim), np.array([])

    # Convert list of arrays to a single numpy array
    return np.array(Xs), np.array(ys)
