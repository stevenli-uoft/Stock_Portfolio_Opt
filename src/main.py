from sklearn.model_selection import cross_val_score

from src.data.DataCollection import StockDataFetcher
from src.data.DataHandling import StockDataHandler
from src.models.StockPricePredictor import RandomForestModel
from src.optimization.MVOModel import PortfolioOptimizer

# def main():
#     fetcher = StockDataFetcher("tests/sample_data")
#     fetcher.fetch_data(start_date="2023-01-01", end_date="2024-01-01")
#     data = fetcher.get_data()
#
#     handler = StockDataHandler(data)
#     handler.clean_data()
#     handler.add_features()


if __name__ == "__main__":
    fetcher = StockDataFetcher("tests/sample_data")
    fetcher.fetch_data(start_date="2023-01-01", end_date="2024-01-01")
    data = fetcher.get_data()

    handler = StockDataHandler(data)
    handler.clean_data()
    handler.add_features()

    predictor = RandomForestModel(data)
    predicted_prices_df = predictor.predict_all()
    # print(predicted_prices_df.head())

    optimizer = PortfolioOptimizer(predicted_prices_df, max_volatility=0.15)
    optimal_weights = optimizer.get_optimal_weights()
    print("Optimal Portfolio Weights:", optimal_weights)
