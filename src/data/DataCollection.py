from fredapi import Fred
import yfinance as yf
import pandas as pd


class StockDataFetcher:
    def __init__(self, csv_path, fred_api_key):
        self.csv_path = csv_path
        self.portfolio = pd.read_csv(self.csv_path)
        self.fred = Fred(api_key=fred_api_key)

    def fetch_stock_data(self, start_date="2023-01-01", end_date="2024-01-01"):
        """ Fetch historical stock data from Yahoo Finance. """
        tickers = self.portfolio['ticker'].tolist()
        ticker_string = " ".join(tickers)
        stock_data = yf.download(ticker_string, start=start_date, end=end_date, group_by='ticker', interval='1wk')
        return stock_data

    def fetch_economic_data(self, start_date="2023-01-01", end_date="2024-01-01"):
        """ Fetch relevant economic data from FRED. """
        economic_indicators = {
            'FRED_T10YIE': 'T10YIE',  # 10-Year Breakeven Inflation Rate (Weekly, Ending Friday)
            'FRED_BAMLH0A0HYM2': 'BAMLH0A0HYM2',  # ICE BofA US High Yield Index Option-Adjusted Spread (Daily)
            'FRED_VIXCLS': 'VIXCLS',  # CBOE Volatility Index: VIX (Daily)
            'FRED_DTWEXBGS': 'DTWEXBGS',  # Trade Weighted U.S. Dollar Index: Broad, Goods (Weekly, Ending Monday)
            'FRED_WPU0561': 'WPU0561',  # Producer Price Index by Commodity: Crude Materials: Crude Petroleum (Monthly)
            'FRED_UMCSENT': 'UMCSENT',  # University of Michigan: Consumer Sentiment (Monthly)
            'FRED_USEPUINDXD': 'USEPUINDXD',  # Economic Policy Uncertainty Index for United States (Daily)
        }

        economic_data = {}
        for name, series_id in economic_indicators.items():
            data = self.fred.get_series(series_id, start_date, end_date)
            # Resample daily data to weekly, forward-filling missing values
            if data.index.freq != 'W':
                data = data.resample('W').last().ffill().bfill()
            economic_data[name] = data

        economic_data = pd.DataFrame(economic_data)
        return economic_data
