import pandas as pd
import yfinance as yf


class StockDataFetcher:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.portfolio = pd.read_csv(self.csv_path)
        self.stock_data = None

    def fetch_data(self, start_date="2020-01-01", end_date="2024-01-01"):
        """ Fetch historical stock data from Yahoo Finance. """
        tickers = self.portfolio['ticker'].tolist()
        ticker_string = " ".join(tickers)  # Create a string of tickers separated by spaces
        data = yf.download(ticker_string, start=start_date, end=end_date, group_by='ticker')
        self.stock_data = data
        return data

    def get_data(self):
        """ Return the fetched stock data. """
        if self.stock_data is not None:
            return self.stock_data
        else:
            raise ValueError("Stock data not fetched. Please run fetch_data() first.")