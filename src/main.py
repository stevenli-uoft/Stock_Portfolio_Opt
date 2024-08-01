from src.data.DataCollection import StockDataFetcher
from src.data.DataHandling import DataHandler
from src.models.StockPricePredictor import RandomForestModel
from src.optimization.MVOModel import PortfolioOptimizer
import pandas as pd

from src.tests.InvestmentSimulator import PortfolioEvaluator

# Constants
FILE_PATH = "tests/sample_data"
FRED_API_KEY = "c9390a380f1c33649e759b91b864853d"

# Risk Levels:
# Low risk: 0.15 (15% annualized volatility)
# Medium risk: 0.25 (25% annualized volatility)
# High risk: 0.35 (35% annualized volatility)
# Very high risk: 0.45 (45% annualized volatility)

# Training date range
TRAINING_START = "2007-01-01"
TRAINING_END = "2021-12-31"

# List of date ranges for analysis periods
date_ranges = [
    {
        "PRODUCTION_START": "2022-01-01",
        "BASELINE_START": "2022-04-01",
        "PRODUCTION_END": "2022-06-30",
        "EVAL_START": "2022-07-01",
        "EVAL_END": "2022-09-30"
    },
    {
        "PRODUCTION_START": "2022-04-01",
        "BASELINE_START": "2022-07-01",
        "PRODUCTION_END": "2022-09-30",
        "EVAL_START": "2022-10-01",
        "EVAL_END": "2022-12-31"
    },
    {
        "PRODUCTION_START": "2022-07-01",
        "BASELINE_START": "2022-10-01",
        "PRODUCTION_END": "2022-12-31",
        "EVAL_START": "2023-01-01",
        "EVAL_END": "2023-03-31"
    },
    {
        "PRODUCTION_START": "2022-10-01",
        "BASELINE_START": "2023-01-01",
        "PRODUCTION_END": "2023-03-31",
        "EVAL_START": "2023-04-01",
        "EVAL_END": "2023-06-30"
    },
    {
        "PRODUCTION_START": "2023-01-01",
        "BASELINE_START": "2023-04-01",
        "PRODUCTION_END": "2023-06-30",
        "EVAL_START": "2023-07-01",
        "EVAL_END": "2023-09-30"
    },
    {
        "PRODUCTION_START": "2023-04-01",
        "BASELINE_START": "2023-07-01",
        "PRODUCTION_END": "2023-09-30",
        "EVAL_START": "2023-10-01",
        "EVAL_END": "2023-12-31"
    },
    {
        "PRODUCTION_START": "2023-07-01",
        "BASELINE_START": "2023-10-01",
        "PRODUCTION_END": "2023-12-31",
        "EVAL_START": "2024-01-01",
        "EVAL_END": "2024-03-31"
    },
    {
        "PRODUCTION_START": "2023-10-01",
        "BASELINE_START": "2024-01-01",
        "PRODUCTION_END": "2024-03-31",
        "EVAL_START": "2024-04-01",
        "EVAL_END": "2024-06-30"
    }
]

if __name__ == "__main__":
    # Collect and handle stock data for training
    fetcher = StockDataFetcher(FILE_PATH, FRED_API_KEY)
    stock_data = fetcher.fetch_stock_data(start_date=TRAINING_START, end_date=TRAINING_END)
    econ_data = fetcher.fetch_economic_data(start_date=TRAINING_START, end_date=TRAINING_END)

    # Flatten the MultiIndex columns in stock_data
    stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]

    # Ensure both dataframes have the same frequency
    stock_data = stock_data.asfreq('W-SUN', method='ffill')
    econ_data = econ_data.asfreq('W-SUN')

    # Align the date ranges
    start_date = max(stock_data.index.min(), econ_data.index.min())
    end_date = min(stock_data.index.max(), econ_data.index.max())

    stock_data = stock_data.loc[start_date:end_date]
    econ_data = econ_data.loc[start_date:end_date]

    # Merge stock and economic data
    master_data = pd.concat([stock_data, econ_data], axis=1)

    # Check if master_data is empty
    if master_data.empty:
        raise ValueError("master_data is empty. Check if stock_data and econ_data have overlapping dates.")

    handler = DataHandler(master_data)
    handler.clean_data()
    master_data = handler.add_features(shift=True)

    # Train and tune the Random Forest Regression Model
    predictor = RandomForestModel(master_data)
    tickers = predictor.get_tickers()
    for ticker in tickers:
        predictor.train_and_evaluate(ticker)

    # Loop through each date range
    for date_range in date_ranges:
        print("\n")
        print("\n=====================================================================")
        print(f"START OF ANALYSIS:")
        print(f"    PRODUCTION PERIOD: {date_range['PRODUCTION_START']} to {date_range['PRODUCTION_END']}")
        print(f"    Evaluation period: {date_range['EVAL_START']} to {date_range['EVAL_END']}")
        print("=====================================================================")

        # Collect production data
        prod_stock_data = fetcher.fetch_stock_data(start_date=date_range['PRODUCTION_START'],
                                                   end_date=date_range['PRODUCTION_END'])
        prod_econ_data = fetcher.fetch_economic_data(start_date=date_range['PRODUCTION_START'],
                                                     end_date=date_range['PRODUCTION_END'])

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

        risk_pref_test = [0.05, 0.15, 0.25, 0.35, 0.45]
        for vol in risk_pref_test:
            print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"Running analysis at {vol} target volatility")
            print(f"Production period: {date_range['PRODUCTION_START']} to {date_range['PRODUCTION_END']}")
            print(f"Evaluation period: {date_range['EVAL_START']} to {date_range['EVAL_END']}")

            # Portfolio Optimization using predicted returns from Random Forest Model
            try:
                optimizer = PortfolioOptimizer(predicted_returns_df, target_volatility=vol)
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
            historical_stock_data = fetcher.fetch_stock_data(start_date=date_range['BASELINE_START'],
                                                             end_date=date_range['PRODUCTION_END'])

            # Flatten the MultiIndex columns in historical_stock_data
            historical_stock_data.columns = ['_'.join(col) for col in historical_stock_data.columns]
            historical_prices_df = historical_stock_data[[col for col in historical_stock_data.columns
                                                          if col.endswith('_Adj Close')]]
            historical_prices_df.columns = [col.split('_')[0] for col in historical_prices_df.columns]

            # Calculate historical returns
            historical_returns_df = historical_prices_df.pct_change(periods=1).dropna()

            # Portfolio Optimization using averaged historical returns
            try:
                baseline_optimizer = PortfolioOptimizer(historical_returns_df, target_volatility=vol)
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
                # Evaluate ML-enhanced portfolio
                ml_evaluator = PortfolioEvaluator(optimal_weights['Weights'])
                ml_return = ml_evaluator.evaluate_portfolio(date_range['EVAL_START'], date_range['EVAL_END'])

                # Evaluate baseline portfolio
                baseline_evaluator = PortfolioEvaluator(baseline_optimal_weights['Weights'])
                baseline_return = baseline_evaluator.evaluate_portfolio(date_range['EVAL_START'],
                                                                        date_range['EVAL_END'])

                # Compare results
                if ml_return is not None and baseline_return is not None:
                    print("\nPortfolio Performance Comparison:")
                    print(f"Evaluation Period: {date_range['EVAL_START']} to {date_range['EVAL_END']}")
                    print(f"RFR Portfolio Return: {optimal_weights['Return']:.4f} ({optimal_weights['Return'] * 100:.2f}%)")
                    print(f"RFR Portfolio Actual Return: {ml_return:.4f} ({ml_return * 100:.2f}%)")
                    print(f"Baseline Portfolio Return: {baseline_optimal_weights['Return']:.4f} ({baseline_optimal_weights['Return'] * 100:.2f}%)")
                    print(f"Baseline Portfolio Actual Return: {baseline_return:.4f} ({baseline_return * 100:.2f}%)")

                    # Calculate relative performance
                    relative_performance = ((1 + ml_return) / (1 + baseline_return) - 1) * 100
                    print(f"Relative Performance: ML-enhanced outperformed baseline by {relative_performance:.2f}%")
            except ValueError:
                print("Optimal weights has not been determined.")

        print("\n=====================================================================")
        print(f"END ANALYSIS OF PRODUCTION PERIOD: {date_range['PRODUCTION_START']} to {date_range['PRODUCTION_END']}")
        print("=====================================================================")
