import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import logging

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
        return X_train, X_test, y_train, y_test

    def optimize_model(self, X_train, y_train):
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
        grid_search.fit(X_train, y_train)
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        logging.info(f'Optimized parameters: {self.best_params}')
        return self.model

    def train_and_evaluate(self, ticker):
        """Train and evaluate the Random Forest model using cross-validation on the training data."""
        X_train, X_test, y_train, y_test = self.prepare_data(ticker)
        self.optimize_model(X_train, y_train)
        # Setup cross-validation on the training data
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
        mse_scores = -cv_scores  # Convert scores to positive MSE scores
        avg_mse = np.mean(mse_scores)
        logging.info(f'Average Cross-validated MSE for {ticker}: {avg_mse}')
        # Final evaluation on the test data
        test_predictions = self.model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_predictions)
        logging.info(f'Test MSE for {ticker}: {test_mse}')
        return test_predictions, test_mse

    def predict_future(self):
        """Predict future prices for all tickers using the test set."""
        predicted_prices = {}
        tickers = self.get_tickers()
        for ticker in tickers:
            logging.info(f'Predicting future prices for ticker: {ticker}')
            predictions, _ = self.train_and_evaluate(ticker)
            predicted_prices[ticker] = predictions
        return pd.DataFrame(predicted_prices,
                            index=self.data.tail(len(predictions)).index)  # Use the index from the test segment

    def get_tickers(self):
        """Extract tickers from the multi-index columns in the DataFrame."""
        return set([col[0] for col in self.data.columns])
