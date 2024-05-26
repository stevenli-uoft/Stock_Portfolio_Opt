import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import logging
import joblib

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RandomForestModel:
    def __init__(self, data):
        self.data = data
        self.models = {}

    def prepare_data(self, ticker, shift_target=True):
        """Prepare data for modeling. Extract features and target for the given ticker."""
        # Extract ticker-specific columns
        ticker_cols = [col for col in self.data.columns if col.startswith(ticker)]
        # Include economic data (columns starting with 'FRED_')
        economic_cols = [col for col in self.data.columns if col.startswith('FRED_')]
        # Combine both ticker and economic columns for features
        feature_cols = ticker_cols + economic_cols

        if shift_target:
            target = self.data[f'{ticker}_Future_Return']  # Predicting future Close price
        else:
            target = self.data[f'{ticker}_Close']  # Directly using Close price for current prediction

        # Drop target features to prevent data leakage
        features = self.data[feature_cols].drop(columns=[f'{ticker}_Future_Return'])

        # Sort the features alphabetically
        features = features[sorted(features.columns)]

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

        return X_train, X_test, y_train, y_test

    def optimize_model(self, X_train, y_train):
        """Optimize Random Forest model using GridSearchCV."""
        parameter_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 20],
            'min_samples_split': [5, 10, 20, 30],
            'min_samples_leaf': [5, 10, 15, 20],
            'max_features': ["sqrt", "log2", None]
        }
        rf = RandomForestRegressor(random_state=42)
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(estimator=rf, param_grid=parameter_grid, cv=tscv, n_jobs=-1, verbose=0,
                                   scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        model = grid_search.best_estimator_
        logging.info(f'Optimized parameters: {best_params}')
        return model, best_params

    def train_and_evaluate(self, ticker):
        """Train and evaluate the Random Forest model using TimeSeriesSplit cross-validation."""
        X_train, X_test, y_train, y_test = self.prepare_data(ticker)
        model, best_params = self.optimize_model(X_train, y_train)
        self.models[ticker] = model  # Save the model for the ticker

        # Setup TimeSeriesSplit cross-validation on the training data
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
        mse_scores = -cv_scores  # Convert scores to positive MSE scores
        avg_mse = np.mean(mse_scores)
        logging.info(f'Average Cross-validated MSE for {ticker}: {avg_mse}')
        # Final evaluation on the test data
        test_predictions = model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_predictions)
        logging.info(f'Test MSE for {ticker}: {test_mse}')
        # Perform feature importance analysis
        self.plot_feature_importances(model, X_train, ticker)
        return test_predictions, test_mse

    def plot_feature_importances(self, model, X_train, ticker):
        """Plot feature importances for the trained model."""
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = X_train.columns

        logging.info("Feature ranking:")
        for i in range(len(importances)):
            logging.info(f"{i + 1}. feature {features[indices[i]]} ({importances[indices[i]]})")

    def predict_future(self, application_data):
        """Predict future prices for all tickers using the application data."""
        predicted_prices = {}
        tickers = self.get_tickers()
        for ticker in tickers:
            logging.info(f'Predicting future prices for ticker: {ticker}')

            # Extract features from application data
            ticker_cols = [col for col in application_data.columns if col.startswith(ticker)]
            economic_cols = [col for col in application_data.columns if col.startswith('FRED_')]
            feature_cols = ticker_cols + economic_cols

            # Sort the features alphabetically
            features = application_data[feature_cols].sort_index(axis=1)

            # Load the trained model
            model = self.models[ticker]

            # Predict the future prices
            predictions = model.predict(features)
            predicted_prices[ticker] = predictions

        # Create a DataFrame with predicted prices
        return pd.DataFrame(predicted_prices, index=application_data.index)

    def get_tickers(self):
        """Extract tickers from the single-level columns in the DataFrame."""
        return set(col.split('_')[0] for col in self.data.columns if '_' in col and not col.startswith('FRED_'))
