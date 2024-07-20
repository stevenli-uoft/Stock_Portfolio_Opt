import pandas as pd
import yfinance as yf


class PortfolioEvaluator:
    def __init__(self, portfolio_weights):
        self.portfolio_weights = portfolio_weights
        self.tickers = list(portfolio_weights.keys())

    def fetch_data(self, start_date, end_date):
        data = yf.download(self.tickers, start=start_date, end=end_date)['Adj Close']
        return data

    def calculate_return(self, start_date, end_date):
        prices = self.fetch_data(start_date, end_date)

        if prices.empty or len(prices) < 2:
            raise ValueError("Insufficient data for the specified date range")

        # Get the starting and ending prices
        start_prices = prices.iloc[0]
        end_prices = prices.iloc[-1]

        # Calculate the return for each stock
        stock_returns = (end_prices - start_prices) / start_prices

        # Calculate the weighted return of the portfolio
        portfolio_return = sum(stock_returns * pd.Series(self.portfolio_weights))

        return portfolio_return

    def evaluate_portfolio(self, start_date, end_date):
        try:
            portfolio_return = self.calculate_return(start_date, end_date)
            return portfolio_return
        except Exception as e:
            print(f"An error occurred during evaluation: {e}")
            return None
