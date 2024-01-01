import pandas as pd
import mysql.connector
from src.config import DB_CONFIG  # Import your database configuration
from sqlalchemy import create_engine


def calculate_features(df):
    """Performs feature engineering on the DataFrame."""
    df['Prev Close'] = df['AdjClose'].shift(1)
    df['SMA_20'] = df['AdjClose'].rolling(window=20).mean()
    df['EMA_20'] = df['AdjClose'].ewm(span=20, adjust=False).mean()
    df['Volume Change'] = df['Volume'].diff()
    df.dropna(inplace=True)
    return df


def fetch_data_and_engineer_features():
    # Using SQLAlchemy engine for connection
    engine = create_engine(
        f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")
    query = "SELECT * FROM StockData"

    try:
        df = pd.read_sql(query, engine)
        df = calculate_features(df)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


# Usage example
processed_data = fetch_data_and_engineer_features()
print(processed_data.head(10))
