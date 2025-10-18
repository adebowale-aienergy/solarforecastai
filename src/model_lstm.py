# src/model_lstm.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path

MODEL_PATH = Path("models/lstm_model.h5")

def load_lstm_model():
    """Load the trained LSTM model."""
    if MODEL_PATH.exists():
        model = load_model(MODEL_PATH)
        return model
    else:
        raise FileNotFoundError(f"LSTM model not found at {MODEL_PATH}")

def predict_lstm(sequence_data: np.ndarray):
    """
    Predict solar power output using the LSTM model.
    Args:
        sequence_data (np.ndarray): Preprocessed 3D input (samples, timesteps, features)
    Returns:
        np.ndarray: Predicted solar output.
    """
    model = load_lstm_model()
    preds = model.predict(sequence_data)
    return preds.flatten()
