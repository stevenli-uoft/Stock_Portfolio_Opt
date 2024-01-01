import pandas as pd
import yfinance as yf
import mysql.connector
from src.config import DB_CONFIG  # Import your database configuration


class YFinanceHandler:
    def __init__(self, user_portfolio_path, start_date='1950-01-02', end_date='2023-12-01'):
        self.connection = None
        self.user_portfolio_path = user_portfolio_path
        self.start_date = start_date
        self.end_date = end_date
        self.additional_tickers = ['^DJI', '^GSPC', '^IXIC', 'VOO', 'VFV.TO', 'VTI', 'VWO', 'AAPL',
                                   'MSFT', 'GOOGL', 'JNJ', 'XOM', 'JPM', 'V', 'PG', 'UNH']

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

    def fetch_and_store_stock_data(self):
        user_portfolio = pd.read_csv(self.user_portfolio_path)
        tickers = user_portfolio['ticker'].tolist() + self.additional_tickers

        cursor = self.connection.cursor()
        insert_query = """
            INSERT INTO StockData (Date, Ticker, AdjClose, Volume)
            VALUES (%s, %s, %s, %s)
        """
        for ticker in tickers:
            ticker_data = yf.download(ticker, start=self.start_date, end=self.end_date)
            for date, row in ticker_data.iterrows():
                try:
                    cursor.execute(insert_query, (date, ticker, row['Adj Close'], row['Volume']))
                except mysql.connector.Error as err:
                    print(f"Error inserting data for {ticker}: {err}")
                    self.connection.rollback()
            print(f"Attempted to store {ticker} into database.")
        self.connection.commit()
        cursor.close()

    def run(self):
        if self.sql_connect():
            self.fetch_and_store_stock_data()
            self.sql_disconnect()
            print("Data fetched and stored successfully")
        else:
            print("failed")


# Example usage
handler = YFinanceHandler('sample_data')
handler.run()
