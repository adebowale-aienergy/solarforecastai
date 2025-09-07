import pandas as pd

def load_dataset(path: str):
    """Load and clean dataset for the app"""
    df = pd.read_csv(path)

    # Ensure standard columns
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"])
    
    return df
