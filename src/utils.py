import pandas as pd
import numpy as np

def preprocess_data(df, date_col, target_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df = df.dropna(subset=[target_col])
    return df

def make_forecast(model, df, horizon, model_type="rf"):
    if model_type == "rf":
        X = np.arange(len(df), len(df) + horizon).reshape(-1, 1)
        preds = model.predict(X)
    elif model_type == "prophet":
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        preds = forecast["yhat"].tail(horizon).values
    elif model_type == "lstm":
        last_seq = df.iloc[-horizon:]["solar_radiation"].values
        preds = model.predict(last_seq.reshape(1, -1, 1))
        preds = preds.flatten()
    else:
        preds = np.zeros(horizon)
    return preds
