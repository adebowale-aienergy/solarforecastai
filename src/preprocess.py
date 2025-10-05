import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_features(df, feature_cols, target_col=None):
    """
    Preprocess dataset by scaling features.
    Returns X (and y if target_col provided).
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])

    if target_col:
        y = df[target_col].values
        return X, y, scaler
    return X, scaler
 
