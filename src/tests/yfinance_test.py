import yfinance as yf


def test_yfinance_download(ticker, start_date, end_date):
    """
    Test function to download and print stock data using yfinance.

    :param ticker: str, the stock ticker symbol.
    :param start_date: str, the start date in 'YYYY-MM-DD' format.
    :param end_date: str, the end date in 'YYYY-MM-DD' format.
    """
    try:
        # Download stock data
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        # Print the first few rows of the data
        print(f"Data for {ticker} from {start_date} to {end_date}:")
        print(stock_data.head())  # Print first few rows
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
test_yfinance_download("VFV", "2022-01-01", "2022-12-31")
