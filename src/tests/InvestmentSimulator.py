import pandas as pd
import yfinance as yf


class PortfolioEvaluator:
    def __init__(self, allocations):
        self.allocations = allocations

    def fetch_data(self, start_date, end_date):
        tickers = list(self.allocations.keys())
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        print(data.head(1))
        print(data.tail(1))
        return data

    def evaluate_performance(self, start_date, end_date):
        data = self.fetch_data(start_date, end_date)
        start_prices = data.iloc[0]
        end_prices = data.iloc[-1]

        # Calculate individual stock returns
        individual_returns = (end_prices - start_prices) / start_prices

        # Calculate portfolio return
        portfolio_return = sum(individual_returns[ticker] * self.allocations[ticker] for ticker in self.allocations)

        return portfolio_return


if __name__ == "__main__":
    allocations = {'MSFT': 0.06163007969463202, 'GOOGL': 0.14202126437609114, 'VFV.TO': 8.900025051520374e-16,
                   'AMZN': 0.01523047596549252, 'NFLX': 2.7405751008412887e-18, 'TSLA': 0.5153290502440645,
                   'META': 0.07634978341321792, 'NVDA': 0.030773078323741564, 'AAPL': 0.15866626798275946}

    evaluator = PortfolioEvaluator(allocations)
    start_date = '2024-01-16'
    end_date = '2024-03-28'
    actual_return = evaluator.evaluate_performance(start_date, end_date)
    print(f"Actual Portfolio Return from {start_date} to {end_date}: {actual_return * 100:.2f}%")
