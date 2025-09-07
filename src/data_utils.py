import pandas as pd
from src.constants import DATA_PATH, DEFAULT_DATE_COL

def load_data():
    df = pd.read_csv(DATA_PATH)
    if DEFAULT_DATE_COL in df.columns:
        df[DEFAULT_DATE_COL] = pd.to_datetime(df[DEFAULT_DATE_COL])
    return df

def filter_by_country(df, country_col, country):
    if country_col in df.columns:
        return df[df[country_col] == country]
    return df
