import joblib
import tensorflow as tf

from src.constants import RF_MODEL_PATH, PROPHET_MODEL_PATH, LSTM_MODEL_PATH

def load_rf_model():
    return joblib.load(RF_MODEL_PATH)

def load_prophet_model():
    return joblib.load(PROPHET_MODEL_PATH)

def load_lstm_model():
    return tf.keras.models.load_model(LSTM_MODEL_PATH)
