import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class StockDataHandler:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()

    def clean_data(self):
        """Clean the stock data by handling missing values and filtering anomalies."""
        # Forward fill and then backward fill the FRED economic data columns
        fred_cols = [col for col in self.data.columns if col.startswith('FRED_')]
        self.data[fred_cols] = self.data[fred_cols].ffill().bfill()

        # Drop rows with any remaining missing values in other columns
        self.data.dropna(inplace=True)

    def add_features(self, short_window=20, long_window=100, ema_span=20):
        """Add multiple features: short and long term moving averages, EMA, volatility, and daily returns in one go,
           and handle NA values created by these operations."""
        for ticker in self.get_tickers():
            # Prepare column names
            close_col = f"{ticker}_Close"
            volume_col = f"{ticker}_Volume"
            future_close_col = f"{ticker}_Future_Close"
            if close_col in self.data.columns:
                close = self.data[close_col]

                # Shift the close prices to create the future target
                self.data[future_close_col] = close.shift(-21)  # Shift by 21 trading days (approximately one month)

                # Short-Term Moving Average
                self.data[f'{ticker}_MA_{short_window}'] = close.rolling(window=short_window).mean()

                # Long-Term Moving Average
                self.data[f'{ticker}_MA_{long_window}'] = close.rolling(window=long_window).mean()

                # Exponential Moving Average
                self.data[f'{ticker}_EMA_{ema_span}'] = close.ewm(span=ema_span, adjust=False).mean()

                # Volatility (standard deviation of daily returns)
                daily_returns = close.pct_change()
                self.data[f'{ticker}_Volatility_{short_window}'] = daily_returns.rolling(window=short_window).std()
                self.data[f'{ticker}_Volatility_{long_window}'] = daily_returns.rolling(window=long_window).std()

                # Daily Returns
                self.data[f'{ticker}_Daily_Returns'] = daily_returns

                # Momentum
                self.data[f'{ticker}_Momentum_{short_window}'] = close - close.shift(short_window)

        # Scale the features
        feature_cols = [col for col in self.data.columns if col not in ['Date'] + list(self.get_tickers())]
        self.data[feature_cols] = self.scaler.fit_transform(self.data[feature_cols])

        # After adding features, clean again to handle NAs produced by rolling calculations
        self.clean_data()

    def get_tickers(self):
        """Extract and return the list of tickers based on the DataFrame's columns."""
        return set(col.split('_')[0] for col in self.data.columns if '_' in col)
