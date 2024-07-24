import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataHandler:
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

    def add_features(self, short_window=4, long_window=12, shift=True):
        new_features = pd.DataFrame(index=self.data.index)

        for ticker in self.get_tickers():
            close_col = f"{ticker}_Adj Close"
            future_return_col = f"{ticker}_Future_Return"
            if close_col in self.data.columns:
                close = self.data[close_col]

                if shift:
                    new_features[future_return_col] = close.pct_change(periods=13).shift(-13)

                # new_features[f'{ticker}_Weekly_Returns'] = close.pct_change()
                new_features[f'{ticker}_MA_{short_window}'] = close.rolling(window=short_window).mean()
                new_features[f'{ticker}_MA_{long_window}'] = close.rolling(window=long_window).mean()

                weekly_returns = close.pct_change()
                # new_features[f'{ticker}_Volatility_{short_window}'] = weekly_returns.rolling(window=short_window).std()
                new_features[f'{ticker}_Volatility_{long_window}'] = weekly_returns.rolling(window=long_window).std()

                # new_features[f'{ticker}_Momentum_{short_window}'] = close.pct_change(periods=short_window)
                new_features[f'{ticker}_Momentum_{long_window}'] = close.pct_change(periods=long_window)

                # Adjusted RSI
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=8).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=8).mean()
                rs = gain / loss
                new_features[f'{ticker}_RSI_8'] = 100 - (100 / (1 + rs))

                # Adjusted MACD
                ema_fast = close.ewm(span=short_window, adjust=False).mean()
                ema_slow = close.ewm(span=long_window, adjust=False).mean()
                new_features[f'{ticker}_MACD'] = ema_fast - ema_slow

                # new_features[f'{ticker}_ROC_{short_window}'] = close.pct_change(periods=short_window)
                new_features[f'{ticker}_ROC_{long_window}'] = close.pct_change(periods=long_window)

        # Concatenate the new features DataFrame with the original data
        self.data = pd.concat([self.data, new_features], axis=1)

        # Scale the features
        feature_cols = [col for col in self.data.columns
                        if col not in ['Date'] + list(self.get_tickers()) +
                        [f'{ticker}_Future_Return' for ticker in self.get_tickers()
                         if f'{ticker}_Future_Return' in self.data.columns]
                        ]
        self.data[feature_cols] = self.scaler.fit_transform(self.data[feature_cols])

        # After adding features, clean again to handle NAs produced by rolling calculations
        self.clean_data()

        return self.data

    def get_tickers(self):
        """Extract and return the list of tickers based on the DataFrame's columns."""
        return set(col.split('_')[0] for col in self.data.columns if '_' in col and not col.startswith('FRED_'))