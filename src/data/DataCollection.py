from fredapi import Fred
import yfinance as yf
import pandas as pd
from typing import Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StockDataFetcher:
    def __init__(self, csv_path: str, fred_api_key: str):
        self.csv_path = csv_path
        self.fred = Fred(api_key=fred_api_key)
        self.portfolio = self._load_portfolio()

    def _load_portfolio(self) -> pd.DataFrame:
        """Load portfolio data from CSV file."""
        try:
            return pd.read_csv(self.csv_path)
        except FileNotFoundError:
            logger.error(f"CSV file not found: {self.csv_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise

    def fetch_stock_data(self, start_date: str = "2023-01-01", end_date: str = "2024-01-01") -> pd.DataFrame:
        """Fetch historical stock data from Yahoo Finance."""
        tickers = self.portfolio['ticker'].tolist()
        ticker_string = " ".join(tickers)
        try:
            stock_data = yf.download(ticker_string, start=start_date, end=end_date, group_by='ticker', interval='1wk')
            logger.info(f"Successfully fetched stock data for {len(tickers)} tickers")
            return stock_data
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            raise

    def fetch_economic_data(self, start_date: str = "2023-01-01", end_date: str = "2024-01-01") -> pd.DataFrame:
        """Fetch relevant economic data from FRED."""
        economic_indicators: Dict[str, str] = {
            'FRED_T10YIE': 'T10YIE',  # 10-Year Breakeven Inflation Rate (Weekly, Ending Friday)
            'FRED_BAMLH0A0HYM2': 'BAMLH0A0HYM2',  # ICE BofA US High Yield Index Option-Adjusted Spread (Daily)
            'FRED_VIXCLS': 'VIXCLS',  # CBOE Volatility Index: VIX (Daily)
            'FRED_DTWEXBGS': 'DTWEXBGS',  # Trade Weighted U.S. Dollar Index: Broad, Goods (Weekly, Ending Monday)
            'FRED_WPU0561': 'WPU0561',  # Producer Price Index by Commodity: Crude Materials: Crude Petroleum (Monthly)
            'FRED_UMCSENT': 'UMCSENT',  # University of Michigan: Consumer Sentiment (Monthly)
            'FRED_USEPUINDXD': 'USEPUINDXD',  # Economic Policy Uncertainty Index for United States (Daily)
        }

        economic_data: Dict[str, pd.Series] = {}
        for name, series_id in economic_indicators.items():
            try:
                data = self.fred.get_series(series_id, start_date, end_date)
                # Resample daily data to weekly, forward-filling missing values
                if data.index.freq != 'W':
                    data = data.resample('W').last().ffill().bfill()
                economic_data[name] = data
                logger.info(f"Successfully fetched data for {name}")
            except Exception as e:
                logger.error(f"Error fetching data for {name}: {str(e)}")

        return pd.DataFrame(economic_data)

    def fetch_all_data(self, start_date: str = "2023-01-01", end_date: str = "2024-01-01") -> pd.DataFrame:
        """Fetch both stock and economic data and merge them."""
        stock_data = self.fetch_stock_data(start_date, end_date)
        economic_data = self.fetch_economic_data(start_date, end_date)

        # Flatten the MultiIndex columns in stock_data
        stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]

        # Merge stock and economic data
        merged_data = pd.merge(stock_data, economic_data, left_index=True, right_index=True, how='outer')
        merged_data.sort_index(inplace=True)

        # Forward fill any missing values
        merged_data.ffill(inplace=True)

        logger.info("Successfully merged stock and economic data")
        return merged_data


if __name__ == "__main__":
    # Example usage
    csv_path = "path/to/your/portfolio.csv"
    fred_api_key = "your_fred_api_key_here"
    start_date = "2023-01-01"
    end_date = "2024-01-01"

    fetcher = StockDataFetcher(csv_path, fred_api_key)
    all_data = fetcher.fetch_all_data(start_date, end_date)
    print(all_data.head())
