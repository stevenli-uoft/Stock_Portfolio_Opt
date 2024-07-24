from src.data.DataCollection import StockDataFetcher
from src.data.DataHandling import DataHandler
from src.models.StockPricePredictor import RandomForestModel
from src.optimization.MVOModel import PortfolioOptimizer
import pandas as pd

from src.tests.InvestmentSimulator import PortfolioEvaluator

if __name__ == "__main__":
    file_path = "tests/sample_data"

    # Collect and handle stock data
    fetcher = StockDataFetcher(file_path, "c9390a380f1c33649e759b91b864853d")
    stock_data = fetcher.fetch_stock_data(start_date="2012-01-01", end_date="2022-12-31")
    econ_data = fetcher.fetch_economic_data(start_date="2012-01-01", end_date="2022-12-31")

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
    prod_start_date = "2022-10-01"  # Extra 3 month as look-back period for lagged features
    prod_end_date = "2023-03-31"
    prod_stock_data = fetcher.fetch_stock_data(start_date=prod_start_date, end_date=prod_end_date)
    prod_econ_data = fetcher.fetch_economic_data(start_date=prod_start_date, end_date=prod_end_date)

    # Flatten the MultiIndex columns in production_stock_data
    prod_stock_data.columns = ['_'.join(col) for col in prod_stock_data.columns]

    # Ensure both dataframes have the same frequency
    prod_stock_data = prod_stock_data.asfreq('W-SUN', method='ffill')
    prod_econ_data = prod_econ_data.asfreq('W-SUN')

    # Align the date ranges
    start_date = max(prod_stock_data.index.min(), prod_econ_data.index.min())
    end_date = min(prod_stock_data.index.max(), prod_econ_data.index.max())

    prod_stock_data = prod_stock_data.loc[start_date:end_date]
    prod_econ_data = prod_econ_data.loc[start_date:end_date]

    prod_master_data = pd.merge(prod_stock_data, prod_econ_data, left_index=True, right_index=True, how='inner')

    prod_data_handler = DataHandler(prod_master_data)
    prod_data_handler.clean_data()
    prod_master_data = prod_data_handler.add_features(shift=False)

    # Predict future returns using the trained models
    predicted_returns_df = predictor.predict_future(prod_master_data)

    # Portfolio Optimization using predicted returns from Random Forest Model
    try:
        optimizer = PortfolioOptimizer(predicted_returns_df, target_volatility=0.35)
        optimal_weights = optimizer.get_optimal_weights()
        print("Predicted Optimal Portfolio Weights:")
        for ticker, weight in optimal_weights['Weights'].items():
            print(f"  {ticker}: {weight:.4f}")
        print("Optimal Portfolio Return:", optimal_weights['Return'])
        print("Optimal Portfolio Volatility:", optimal_weights['Volatility'])
        print("Optimal Portfolio Sharpe Ratio:", optimal_weights['Sharpe Ratio'])
    except Exception as e:
        print(f"An error occurred during portfolio optimization: {e}")

    # Baseline MVO Model using historical returns
    baseline_start_date = "2023-01-01"
    baseline_end_date = "2023-03-31"
    historical_stock_data = fetcher.fetch_stock_data(start_date=baseline_start_date, end_date=baseline_end_date)

    # Flatten the MultiIndex columns in historical_stock_data
    historical_stock_data.columns = ['_'.join(col) for col in historical_stock_data.columns]
    historical_prices_df = historical_stock_data[[col for col in historical_stock_data.columns
                                                  if col.endswith('_Adj Close')]]
    historical_prices_df.columns = [col.split('_')[0] for col in historical_prices_df.columns]

    # Calculate historical returns
    historical_returns_df = historical_prices_df.pct_change(periods=1).dropna()

    # Portfolio Optimization using averaged historical returns
    try:
        baseline_optimizer = PortfolioOptimizer(historical_returns_df, target_volatility=0.35)
        baseline_optimal_weights = baseline_optimizer.get_optimal_weights()
        print("Baseline Optimal Portfolio Weights:")
        for ticker, weight in baseline_optimal_weights['Weights'].items():
            print(f"  {ticker}: {weight:.4f}")
        print("Baseline Optimal Portfolio Return:", baseline_optimal_weights['Return'])
        print("Baseline Optimal Portfolio Volatility:", baseline_optimal_weights['Volatility'])
        print("Baseline Optimal Portfolio Sharpe Ratio:", baseline_optimal_weights['Sharpe Ratio'])
    except Exception as e:
        print(f"An error occurred during baseline portfolio optimization: {e}")

    try:
        # Set evaluation period (adjust as needed)
        eval_start_date = "2023-04-01"
        eval_end_date = "2023-06-30"

        # Evaluate ML-enhanced portfolio
        ml_evaluator = PortfolioEvaluator(optimal_weights['Weights'])
        ml_return = ml_evaluator.evaluate_portfolio(eval_start_date, eval_end_date)

        # Evaluate baseline portfolio
        baseline_evaluator = PortfolioEvaluator(baseline_optimal_weights['Weights'])
        baseline_return = baseline_evaluator.evaluate_portfolio(eval_start_date, eval_end_date)

        # Compare results
        if ml_return is not None and baseline_return is not None:
            print("\nPortfolio Performance Comparison:")
            print(f"Evaluation Period: {eval_start_date} to {eval_end_date}")
            print(f"RFR Portfolio Return: {optimal_weights['Return']:.4f} ({optimal_weights['Return'] * 100:.2f}%)")
            print(f"RFR Portfolio Actual Return: {ml_return:.4f} ({ml_return * 100:.2f}%)")
            print(f"Baseline Portfolio Return: {baseline_optimal_weights['Return']:.4f} ({baseline_optimal_weights['Return'] * 100:.2f}%)")
            print(f"Baseline Portfolio Actual Return: {baseline_return:.4f} ({baseline_return * 100:.2f}%)")

            # Calculate relative performance
            relative_performance = ((1 + ml_return) / (1 + baseline_return) - 1) * 100
            print(f"Relative Performance: ML-enhanced outperformed baseline by {relative_performance:.2f}%")
    except ValueError:
        print("Optimal weights has not been determined.")
