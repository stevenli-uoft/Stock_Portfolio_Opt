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
    allocations = {'MSFT': 0.0, 'AAPL': 9.322792424702106e-16, 'VFV.TO': 4.95473227694169e-16,
                   'GOOGL': 0.21890883215046894, 'NFLX': 0.42167636626372723, 'TSLA': 0.28382925313290774,
                   'META': 4.0075965359051273e-16, 'NVDA': 0.0755855484528945, 'AMZN': 1.5655231422342987e-16}

    evaluator = PortfolioEvaluator(allocations)
    start_date = '2023-06-01'
    end_date = '2024-05-25'
    actual_return = evaluator.evaluate_performance(start_date, end_date)
    print(f"Actual Portfolio Return from {start_date} to {end_date}: {actual_return * 100:.2f}%")
