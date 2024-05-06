import pandas as pd
import numpy as np

from src.data.DataCollection import StockDataFetcher


class StockDataHandler:
    def __init__(self, data):
        self.data = data

    def clean_data(self):
        """ Clean the stock data by handling missing values and filtering anomalies. """
        # Forward fill any missing values
        self.data.ffill(inplace=True)
        # Backward fill any remaining missing values
        self.data.bfill(inplace=True)
        return self.data

    def add_features(self, window=20, ema_span=20):
        """ Add multiple features: moving average, EMA, volatility, and daily returns in one go,
            and handle NA values created by these operations. """
        for ticker in self.get_tickers():
            ticker_data = self.data[ticker]
            close = ticker_data['Close']
            # Moving Average
            self.data[ticker, 'MA_' + str(window)] = close.rolling(window=window).mean()
            # Exponential Moving Average
            self.data[ticker, 'EMA_' + str(ema_span)] = close.ewm(span=ema_span, adjust=False).mean()
            # Volatility (standard deviation of daily returns)
            daily_returns = close.pct_change()
            self.data[ticker, 'Volatility_' + str(window)] = daily_returns.rolling(window=window).std()
            # Daily Returns
            self.data[ticker, 'Daily_Returns'] = daily_returns

        # After adding features, clean again to handle NAs produced by rolling calculations
        self.clean_features()

    def clean_features(self):
        """ Clean the engineered features by forward filling and then backward filling to handle NAs. """
        # Specifically targeting newly added columns that could have NAs
        for ticker in self.get_tickers():
            for feature in ['MA_20', 'EMA_20', 'Volatility_20', 'Daily_Returns']:
                if (ticker, feature) in self.data.columns:
                    self.data[ticker, feature].ffill(inplace=True)
                    self.data[ticker, feature].bfill(inplace=True)

    def get_tickers(self):
        """ Extract and return the list of tickers based on the DataFrame's columns. """
        return [item[0] for item in set(self.data.columns) if isinstance(item, tuple)]

