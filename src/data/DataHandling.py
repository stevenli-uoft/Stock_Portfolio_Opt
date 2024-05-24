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

    def add_features(self, short_window=20, long_window=50, ema_span=20):
        """Add multiple features: short and long term moving averages, EMA, volatility, daily returns, RSI, MACD,
           Bollinger Bands, rolling skewness, and rolling kurtosis in one go, and handle NA values created by these operations."""
        new_features = pd.DataFrame(index=self.data.index)

        for ticker in self.get_tickers():
            # Prepare column names
            close_col = f"{ticker}_Close"
            future_close_col = f"{ticker}_Future_Close"
            if close_col in self.data.columns:
                close = self.data[close_col]

                # Shift the close prices to create the future target
                new_features[future_close_col] = close.shift(-21)  # Shift by 21 trading days (approximately one month)

                # Moving Averages
                new_features[f'{ticker}_MA_{short_window}'] = close.rolling(window=short_window).mean()
                new_features[f'{ticker}_MA_{long_window}'] = close.rolling(window=long_window).mean()
                new_features[f'{ticker}_EMA_{ema_span}'] = close.ewm(span=ema_span, adjust=False).mean()

                # Volatility
                daily_returns = close.pct_change()
                new_features[f'{ticker}_Volatility_{short_window}'] = daily_returns.rolling(window=short_window).std()

                # Daily Returns
                new_features[f'{ticker}_Daily_Returns'] = daily_returns

                # Lagged Returns
                new_features[f'{ticker}_5d_Lagged_Return'] = daily_returns.shift(5)
                new_features[f'{ticker}_10d_Lagged_Return'] = daily_returns.shift(10)
                new_features[f'{ticker}_21d_Lagged_Return'] = daily_returns.shift(21)

                # Relative Strength Index (RSI)
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                new_features[f'{ticker}_RSI'] = 100 - (100 / (1 + rs))

                # Bollinger Bands
                new_features[f'{ticker}_Bollinger_Upper'] = (
                        new_features[f'{ticker}_MA_{short_window}'] +
                        (2 * new_features[f'{ticker}_Volatility_{short_window}']))
                new_features[f'{ticker}_Bollinger_Lower'] = (
                        new_features[f'{ticker}_MA_{short_window}'] -
                        (2 * new_features[f'{ticker}_Volatility_{short_window}']))

                # MACD
                ema_12 = close.ewm(span=12, adjust=False).mean()
                ema_26 = close.ewm(span=26, adjust=False).mean()
                macd = ema_12 - ema_26
                signal = macd.ewm(span=9, adjust=False).mean()
                new_features[f"{ticker}_MACD"] = macd
                new_features[f"{ticker}_MACD_Signal"] = signal

        # Concatenate the new features DataFrame with the original data
        self.data = pd.concat([self.data, new_features], axis=1)

        # Scale the features
        feature_cols = [col for col in self.data.columns if col not in ['Date'] + list(self.get_tickers())]
        self.data[feature_cols] = self.scaler.fit_transform(self.data[feature_cols])

        # After adding features, clean again to handle NAs produced by rolling calculations
        self.clean_data()

        return self.data

    def get_tickers(self):
        """Extract and return the list of tickers based on the DataFrame's columns."""
        return set(col.split('_')[0] for col in self.data.columns if '_' in col)
