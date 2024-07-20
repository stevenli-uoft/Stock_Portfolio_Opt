import pandas as pd
import yfinance as yf
from fredapi import Fred


class StockDataFetcher:
    def __init__(self, csv_path, fred_api_key):
        self.csv_path = csv_path
        self.portfolio = pd.read_csv(self.csv_path)
        self.fred = Fred(api_key=fred_api_key)

    def fetch_stock_data(self, start_date="2023-01-01", end_date="2024-01-01"):
        """ Fetch historical stock data from Yahoo Finance. """
        tickers = self.portfolio['ticker'].tolist()
        ticker_string = " ".join(tickers)  # Create a string of tickers separated by spaces
        stock_data = yf.download(ticker_string, start=start_date, end=end_date, group_by='ticker')
        return stock_data

    def fetch_economic_data(self, start_date="2023-01-01", end_date="2024-01-01"):
        """ Fetch relevant economic data from FRED. """
        # List of economic data series you want to fetch
        economic_indicators = {
            'FRED_DTB3': 'DTB3',  # 3-Month Treasury Bill: Secondary Market Rate
            'FRED_DGS10': 'DGS10',  # 10-Year Treasury Constant Maturity Rate
            'FRED_SP500': 'SP500',  # S&P 500 Index
            'FRED_OILPRICE': 'DCOILWTICO',  # Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma
            'FRED_CPIAUCSL': 'CPIAUCSL',  # Consumer Price Index (Monthly)
            'FRED_PCEPILFE': 'PCEPILFE'  # PCE Price Index Excluding Food and Energy (Monthly)
        }

        economic_data = {}
        for name, series_id in economic_indicators.items():
            economic_data[name] = self.fred.get_series(series_id, start_date, end_date)

        economic_data = pd.DataFrame(economic_data)
        return economic_data
