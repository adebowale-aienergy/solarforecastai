import datetime

def format_date(date_str):
    """Format date string into datetime object."""
    return datetime.datetime.strptime(date_str, "%Y-%m-%d")

def select_country(df, country_name, col="country"):
    """Filter dataset for a specific country."""
    return df[df[col] == country_name]
 
