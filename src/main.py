from src.data.DataCollection import StockDataFetcher
from src.data.DataHandling import DataHandler
from src.models.StockPricePredictor import RandomForestModel
from src.optimization.MVOModel import PortfolioOptimizer
import pandas as pd

if __name__ == "__main__":
    file_path = "tests/sample_data"

    # Collect and handle stock data
    fetcher = StockDataFetcher(file_path, "c9390a380f1c33649e759b91b864853d")
    fetcher.fetch_stock_data(start_date="2000-01-01", end_date="2022-05-31")
    stock_data = fetcher.get_stock_data()
    fetcher.fetch_economic_data(start_date="2000-01-01", end_date="2022-05-31")
    econ_data = fetcher.get_economic_data()

    # Flatten the MultiIndex columns in stock_data
    stock_data.columns = ['_'.join(col) for col in stock_data.columns]

    master_data = pd.merge(stock_data, econ_data, left_index=True, right_index=True, how='inner')

    handler = DataHandler(master_data)
    handler.clean_data()
    master_data = handler.add_features(shift=True)

    # Train and tune the Random Forest Regression Model
    predictor = RandomForestModel(master_data)
    tickers = predictor.get_tickers()
    for ticker in tickers:
        predictor.train_and_evaluate(ticker)

    # Collect production data
    prod_start_date = "2022-06-01"
    prod_end_date = "2023-05-31"
    fetcher.fetch_stock_data(start_date=prod_start_date, end_date=prod_end_date)
    prod_stock_data = fetcher.get_stock_data()
    fetcher.fetch_economic_data(start_date=prod_start_date, end_date=prod_end_date)
    prod_econ_data = fetcher.get_economic_data()

    # Flatten the MultiIndex columns in production_stock_data
    prod_stock_data.columns = ['_'.join(col) for col in prod_stock_data.columns]

    prod_master_data = pd.merge(prod_stock_data, prod_econ_data, left_index=True, right_index=True, how='inner')

    prod_data_handler = DataHandler(prod_master_data)
    prod_data_handler.clean_data()
    prod_master_data = prod_data_handler.add_features(shift=False)

    # Predict future returns using the trained models
    predicted_returns_df = predictor.predict_future(prod_master_data)

    # Portfolio Optimization with MVOModel
    try:
        optimizer = PortfolioOptimizer(predicted_returns_df, max_volatility=0.2)
        optimal_weights = optimizer.get_optimal_weights()
        print("Optimal Portfolio Weights:", optimal_weights['Weights'])
        print("Optimal Portfolio Return:", optimal_weights['Return'])
        print("Optimal Portfolio Volatility:", optimal_weights['Volatility'])
        print("Optimal Portfolio Sharpe Ratio:", optimal_weights['Sharpe Ratio'])
    except Exception as e:
        print(f"An error occurred during portfolio optimization: {e}")

    # Baseline MVO Model using historical returns
    fetcher.fetch_stock_data(start_date=prod_start_date, end_date=prod_end_date)
    historical_stock_data = fetcher.get_stock_data()

    # Flatten the MultiIndex columns in historical_stock_data
    historical_stock_data.columns = ['_'.join(col) for col in historical_stock_data.columns]

    historical_prices_df = historical_stock_data[[col for col in historical_stock_data.columns
                                                  if col.endswith('_Close')]]

    # Calculate historical returns
    historical_returns_df = historical_prices_df.pct_change().dropna()

    # Portfolio Optimization with the baseline MVO model
    try:
        baseline_optimizer = PortfolioOptimizer(historical_returns_df, max_volatility=0.2)
        baseline_optimal_weights = baseline_optimizer.get_optimal_weights()
        print("Baseline Optimal Portfolio Weights:", baseline_optimal_weights['Weights'])
        print("Baseline Optimal Portfolio Return:", baseline_optimal_weights['Return'])
        print("Baseline Optimal Portfolio Volatility:", baseline_optimal_weights['Volatility'])
        print("Baseline Optimal Portfolio Sharpe Ratio:", baseline_optimal_weights['Sharpe Ratio'])
    except Exception as e:
        print(f"An error occurred during baseline portfolio optimization: {e}")
