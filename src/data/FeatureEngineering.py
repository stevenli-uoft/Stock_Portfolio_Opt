import pandas as pd
import mysql.connector
from src.config import DB_CONFIG  # Import your database configuration


def calculate_features(df):
    """Performs feature engineering on the DataFrame."""
    df['Prev Close'] = df['AdjClose'].shift(1)
    df['SMA_20'] = df['AdjClose'].rolling(window=20).mean()
    df['EMA_20'] = df['AdjClose'].ewm(span=20, adjust=False).mean()
    df['Volume Change'] = df['Volume'].diff()
    df.dropna(inplace=True)
    return df


class FeatureEngineeringHandler:
    def __init__(self):
        self.connection = None

    def sql_connect(self):
        """Establishes a connection to the database."""
        try:
            self.connection = mysql.connector.connect(
                host=DB_CONFIG['host'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password'],
                database=DB_CONFIG['database']
            )
            return True
        except mysql.connector.Error as err:
            print(f"Database connection error: {err}")
            return False

    def sql_disconnect(self):
        """Closes the database connection."""
        if self.connection:
            self.connection.close()

    def fetch_data_and_engineer_features(self):
        if not self.sql_connect():
            return None

        query = "SELECT * FROM StockData"  # Adjust this query based on your actual table structure
        try:
            df = pd.read_sql(query, self.connection)
            df = calculate_features(df)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
        finally:
            self.sql_disconnect()


# Usage example
# handler = FeatureEngineeringHandler()
# processed_data = handler.fetch_data_and_engineer_features()
# print(processed_data.head(10))
