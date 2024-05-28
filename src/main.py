from src.data.DataCollection import StockDataFetcher
from src.data.DataHandling import StockDataHandler
from src.models.StockPricePredictor import RandomForestModel
from src.optimization.MVOModel import PortfolioOptimizer
import joblib
import pandas as pd

if __name__ == "__main__":
    file_path = "tests/sample_data"
    fetcher = StockDataFetcher(file_path, "c9390a380f1c33649e759b91b864853d")

    # Collect and handle stock data
    fetcher.fetch_stock_data(start_date="2000-01-01", end_date="2023-12-31")
    stock_data = fetcher.get_stock_data()
    fetcher.fetch_economic_data(start_date="2000-01-01", end_date="2023-12-31")
    econ_data = fetcher.get_economic_data()

    # Flatten the MultiIndex columns in stock_data
    stock_data.columns = ['_'.join(col) for col in stock_data.columns]

    master_data = pd.merge(stock_data, econ_data, left_index=True, right_index=True, how='inner')

    handler = StockDataHandler(master_data)
    handler.clean_data()
    master_data = handler.add_features(shift=True)

    # Train and tune the Random Forest Regression Model
    predictor = RandomForestModel(master_data)
    tickers = predictor.get_tickers()
    for ticker in tickers:
        predictor.train_and_evaluate(ticker)

    # Save the models for future use
    for ticker in tickers:
        joblib.dump(predictor.models[ticker], f"models/{ticker}_model.pkl")

    # Collect application data
    application_start_date = "2023-10-15"
    application_end_date = "2024-03-31"  # Adjust this as needed
    fetcher.fetch_stock_data(start_date=application_start_date, end_date=application_end_date)
    application_stock_data = fetcher.get_stock_data()
    fetcher.fetch_economic_data(start_date=application_start_date, end_date=application_end_date)
    application_econ_data = fetcher.get_economic_data()

    # Flatten the MultiIndex columns in application_stock_data
    application_stock_data.columns = ['_'.join(col) for col in application_stock_data.columns]

    application_data = pd.merge(application_stock_data, application_econ_data,
                                left_index=True, right_index=True, how='inner')

    application_handler = StockDataHandler(application_data)
    application_handler.clean_data()
    application_data = application_handler.add_features(shift=False)

    # Load the models for each ticker and predict future prices
    for ticker in tickers:
        predictor.models[ticker] = joblib.load(f"models/{ticker}_model.pkl")

    predicted_prices_df = predictor.predict_future(application_data)

    # Portfolio Optimization with MVOModel
    optimizer = PortfolioOptimizer(predicted_prices_df, max_volatility=0.8)
    optimal_weights = optimizer.get_optimal_weights()
    print("Optimal Portfolio Weights:", optimal_weights['Weights'])
    print("Optimal Portfolio Return:", optimal_weights['Return'])
    print("Optimal Portfolio Volatility:", optimal_weights['Volatility'])
    print("Optimal Portfolio Sharpe Ratio:", optimal_weights['Sharpe Ratio'])
