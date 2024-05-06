from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


class RandomForestModel:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.best_params = None

    def prepare_data(self, ticker):
        """Prepare data for modeling. Extract features and target for the given ticker."""
        features = self.data.loc[:, (ticker, slice(None))]
        features = features.drop(columns=[(ticker, 'Close'), (ticker, 'Adj Close')])  # Drop target columns
        target = self.data[(ticker, 'Close')]  # We'll predict the Close price
        return features, target

    def optimize_model(self, features, target):
        """Optimize Random Forest model using GridSearchCV."""
        parameter_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=parameter_grid, cv=3, n_jobs=-1, verbose=0,
                                   scoring='neg_mean_squared_error')
        grid_search.fit(features, target)
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_

    def train_and_evaluate(self, ticker):
        """Train and evaluate the Random Forest model using cross-validation."""
        features, target = self.prepare_data(ticker)
        self.optimize_model(features, target)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, features, target, cv=kfold, scoring='neg_mean_squared_error')
        mse_scores = -cv_scores  # Convert scores to positive MSE scores
        avg_mse = np.mean(mse_scores)
        print(f'Cross-validated MSE for {ticker}: {avg_mse}')
        return avg_mse

    def predict_all(self):
        """Predict prices for all tickers in the dataset, including the date for each prediction."""
        predicted_prices = {}
        tickers = self.get_tickers()
        for ticker in tickers:
            self.train_and_evaluate(ticker)
            all_features = self.data.loc[:, (ticker, slice(None))]
            all_features = all_features.drop(columns=[(ticker, 'Close'), (ticker, 'Adj Close')])
            predicted_prices[ticker] = self.model.predict(all_features)

        # Create DataFrame with dates and predicted prices for each ticker
        date_index = self.data.index  # Assuming the index of your DataFrame is the date
        predicted_df = pd.DataFrame(predicted_prices, index=date_index)
        return predicted_df

    def get_tickers(self):
        """Extract tickers from the multi-index columns in the DataFrame."""
        return set([col[0] for col in self.data.columns])