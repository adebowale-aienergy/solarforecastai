import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from prophet.serialize import model_from_json

def load_models(rf_path: str, prophet_path: str, lstm_path: str):
    """Load trained models (Random Forest, Prophet, LSTM)."""
    models = {}

    # Random Forest
    try:
        models["rf"] = joblib.load(rf_path)
    except Exception as e:
        print(f"Random Forest model not loaded: {e}")

    # Prophet
    try:
        with open(prophet_path, "r") as f:
            models["prophet"] = model_from_json(f.read())
    except Exception as e:
        print(f"Prophet model not loaded: {e}")

    # LSTM
    try:
        models["lstm"] = load_model(lstm_path)
    except Exception as e:
        print(f"LSTM model not loaded: {e}")

    return models


def make_prediction(model, model_type: str, X, horizon: int = 7):
    """
    Generate predictions with the selected model.
    - model_type: "rf", "prophet", or "lstm"
    """
    if model_type == "rf":
        return model.predict(X)

    elif model_type == "prophet":
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        return forecast[["ds", "yhat"]].tail(horizon)

    elif model_type == "lstm":
        preds = model.predict(X)
        return preds

    else:
        raise ValueError(f"Unknown model type: {model_type}")
