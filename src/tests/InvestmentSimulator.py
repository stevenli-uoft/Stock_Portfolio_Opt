import pandas as pd
import yfinance as yf


class PortfolioEvaluator:
    def __init__(self, allocations):
        self.allocations = allocations

    def fetch_data(self, start_date, end_date):
        tickers = list(self.allocations.keys())
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        return data

    def calculate_returns(self, data):
        returns = data.pct_change().dropna()
        return returns

    def evaluate_performance(self, start_date, end_date):
        data = self.fetch_data(start_date, end_date)
        returns = self.calculate_returns(data)

        # Calculate portfolio return
        portfolio_returns = returns.dot(pd.Series(self.allocations))

        # Calculate cumulative return
        cumulative_return = (1 + portfolio_returns).prod() - 1

        return cumulative_return


if __name__ == "__main__":
    allocations = {'META': 0.03850368597352959, 'TSLA': 5.92668953193834e-17, 'AMZN': 0.02164008391651878,
                   'GOOGL': 0.3145029720748397, 'MSFT': 0.06532087274653729, 'NFLX': 0.11741451186072589,
                   'VFV.TO': 0.15854420758054213, 'AAPL': 3.906515952736833e-18, 'NVDA': 0.28407366584730653}

    evaluator = PortfolioEvaluator(allocations)
    start_date = '2024-03-31'
    end_date = '2024-04-30'
    actual_return = evaluator.evaluate_performance(start_date, end_date)
    print(f"Actual Portfolio Return from {start_date} to {end_date}: {actual_return * 100:.2f}%")
