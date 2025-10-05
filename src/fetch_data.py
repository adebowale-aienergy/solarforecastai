import os
from huggingface_hub import hf_hub_download
import pandas as pd
import joblib
import tensorflow as tf

# Your Hugging Face repo (datasets)
HF_REPO = "adebowale-aienergy/solarforecastai"

# ----------- MODELS -----------
def load_random_forest():
    model_path = hf_hub_download(repo_id=HF_REPO, filename="models/random_forest_model.pkl")
    return joblib.load(model_path)

def load_prophet():
    model_path = hf_hub_download(repo_id=HF_REPO, filename="models/prophet_model.pkl")
    from prophet import Prophet
    return joblib.load(model_path)

def load_lstm():
    model_path = hf_hub_download(repo_id=HF_REPO, filename="models/LSTM_model.h5")
    return tf.keras.models.load_model(model_path)


# ----------- DATA -----------
def load_clean_data():
    file_path = hf_hub_download(repo_id=HF_REPO, filename="data/processed/clean_data.csv")
    return pd.read_csv(file_path)

def load_features_data():
    file_path = hf_hub_download(repo_id=HF_REPO, filename="data/processed/features_data.csv")
    return pd.read_csv(file_path)

def load_raw_cams():
    file_path = hf_hub_download(repo_id=HF_REPO, filename="data/raw/cams_for_32_countries.csv")
    return pd.read_csv(file_path)

def load_raw_nasa():
    file_path = hf_hub_download(repo_id=HF_REPO, filename="data/raw/nasa_power_global_32countries.csv")
    return pd.read_csv(file_path)

def load_merged_data():
    file_path = hf_hub_download(repo_id=HF_REPO, filename="data/merged/comprehensive_solar_data_hourly.csv")
    return pd.read_csv(file_path)
 
