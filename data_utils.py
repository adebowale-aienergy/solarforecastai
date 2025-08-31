import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.
    Args:
        filepath (str): Path to the dataset.
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    df = pd.read_csv(filepath)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df
