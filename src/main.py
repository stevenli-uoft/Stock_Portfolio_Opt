from src.data.DataCollection import StockDataFetcher
from src.data.DataHandling import StockDataHandler
from src.models.StockPricePredictor import RandomForestModel
from src.optimization.MVOModel import PortfolioOptimizer

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

if __name__ == "__main__":
    file_path = "tests/sample_data"

    # Collect and handle stock data
    fetcher = StockDataFetcher(file_path, "c9390a380f1c33649e759b91b864853d")
    fetcher.fetch_stock_data(start_date="2000-01-01", end_date="2024-01-01")
    stock_data = fetcher.get_stock_data()
    fetcher.fetch_economic_data(start_date="2000-01-01", end_date="2024-01-01")
    econ_data = fetcher.get_economic_data()

    # Flatten the MultiIndex columns in stock_data
    stock_data.columns = ['_'.join(col) for col in stock_data.columns]

    master_data = pd.merge(stock_data, econ_data, left_index=True, right_index=True, how='inner')

    handler = StockDataHandler(master_data)
    handler.clean_data()
    handler.add_features()

    # Initialize and use the Random Forest Regression Model
    predictor = RandomForestModel(master_data)
    predicted_prices_df = predictor.predict_future()
    # print(predicted_prices_df.head())

    # Portfolio Optimization with MVOModel
    optimizer = PortfolioOptimizer(predicted_prices_df, max_volatility=0.15)
    optimal_weights = optimizer.get_optimal_weights()
    print("Optimal Portfolio Weights:", optimal_weights['Weights'])
    print("Optimal Portfolio Return:", optimal_weights['Return'])
    print("Optimal Portfolio Volatility:", optimal_weights['Volatility'])
    print("Optimal Portfolio Sharpe Ratio:", optimal_weights['Sharpe Ratio'])