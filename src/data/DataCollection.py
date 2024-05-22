import pandas as pd
import yfinance as yf
from fredapi import Fred


class StockDataFetcher:
    def __init__(self, csv_path, fred_api_key):
        self.csv_path = csv_path
        self.portfolio = pd.read_csv(self.csv_path)
        self.stock_data = None
        self.benchmark_data = None
        self.fred = Fred(api_key=fred_api_key)
        self.economic_data = None

    def fetch_stock_data(self, start_date="2020-01-01", end_date="2024-01-01"):
        """ Fetch historical stock data from Yahoo Finance. """
        tickers = self.portfolio['ticker'].tolist()
        ticker_string = " ".join(tickers)  # Create a string of tickers separated by spaces
        data = yf.download(ticker_string, start=start_date, end=end_date, group_by='ticker')
        self.stock_data = data
        return data

    def fetch_economic_data(self, start_date="2000-01-01", end_date="2024-01-01"):
        """ Fetch relevant economic data from FRED. """
        # List of economic data series you want to fetch
        economic_indicators = {
            'FRED_DTB3': 'DTB3',  # 3-Month Treasury Bill: Secondary Market Rate
            'FRED_DGS10': 'DGS10',  # 10-Year Treasury Constant Maturity Rate
            'FRED_SP500': 'SP500',  # S&P 500 Index
            'FRED_OILPRICE': 'DCOILWTICO',  # Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma
            'FRED_EURUSD': 'DEXUSEU',  # U.S. / Euro Foreign Exchange Rate
            'FRED_VIXCLS': 'VIXCLS',  # CBOE Volatility Index: VIX
            'FRED_CPIAUCSL': 'CPIAUCSL',  # Consumer Price Index (Monthly)
            'FRED_PCEPILFE': 'PCEPILFE'  # PCE Price Index Excluding Food and Energy (Monthly)
        }

        economic_data = {}
        for name, series_id in economic_indicators.items():
            economic_data[name] = self.fred.get_series(series_id, start_date, end_date)

        self.economic_data = pd.DataFrame(economic_data)
        return self.economic_data

    def get_stock_data(self):
        """ Return the fetched stock data. """
        if self.stock_data is not None:
            return self.stock_data
        else:
            raise ValueError("Stock data not fetched. Please run fetch_stock_data() first.")

    def get_economic_data(self):
        """ Return the fetched economic data. """
        if self.economic_data is not None:
            return self.economic_data
        else:
            raise ValueError("Economic data not fetched. Please run fetch_economic_data() first.")
