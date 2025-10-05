# utils.py
import os
import joblib
import pickle
import tensorflow as tf

# ---------------------------
# Model Save / Load Helpers
# ---------------------------

def save_model(model, path, model_type="sklearn"):
    """
    Save a model to disk.
    
    Args:
        model: trained model
        path: save path
        model_type: "sklearn", "prophet", "lstm"
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if model_type in ["sklearn", "prophet"]:
        joblib.dump(model, path)
    elif model_type == "lstm":
        model.save(path)  # saves in .h5
    else:
        raise ValueError("Unknown model type")


def load_model(path, model_type="sklearn"):
    """
    Load a saved model from disk.
    """
    if model_type in ["sklearn", "prophet"]:
        return joblib.load(path)
    elif model_type == "lstm":
        return tf.keras.models.load_model(path)
    else:
        raise ValueError("Unknown model type")


# ---------------------------
# Data Helpers
# ---------------------------

def split_data(df, target_col, test_size=0.2):
    """
    Splits dataframe into train/test sets.
    """
    from sklearn.model_selection import train_test_split
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=42)
