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
    allocations = {'JPM_Close': 0.0, 'MSFT_Close': 1.7867651802561113e-16, 'AMZN_Close': 0.0,
                   'XOM_Close': 0.0, 'WMT_Close': 1.362592012888616e-16, 'GOOGL_Close': 1.5561377074248253e-16,
                   'JNJ_Close': 0.0, 'NFLX_Close': 0.4530310365224935, 'PG_Close': 0.0,
                   'AAPL_Close': 1.206699739390293e-16, 'PFE_Close': 0.0, 'NVDA_Close': 0.5469689634775063,
                   'V_Close': 0.0, 'TSLA_Close': 0.0, 'DIS_Close': 0.0}

    evaluator = PortfolioEvaluator(allocations)
    start_date = '2022-06-02'
    end_date = '2023-05-30'
    actual_return = evaluator.evaluate_performance(start_date, end_date)
    print(f"Actual Portfolio Return from {start_date} to {end_date}: {actual_return * 100:.2f}%")
