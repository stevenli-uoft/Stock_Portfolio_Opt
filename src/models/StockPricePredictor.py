import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import joblib

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RandomForestModel:
    def __init__(self, data):
        self.data = data
        self.models = {}

    def prepare_data(self, ticker):
        """Prepare data for modeling. Extract features and target for the given ticker."""
        # Extract ticker-specific columns
        ticker_cols = [col for col in self.data.columns if col.startswith(ticker)]

        # Include economic data (columns starting with 'FRED_')
        economic_cols = [col for col in self.data.columns if col.startswith('FRED_')]

        # Combine both ticker and economic columns for features
        feature_cols = ticker_cols + economic_cols

        # Target for predicting future returns
        target = self.data[f'{ticker}_Future_Return']

        # Drop target features to prevent data leakage, and drop un-important features
        features = self.data[feature_cols].drop(columns=[f'{ticker}_Future_Return'])

        # Sort the features alphabetically
        features = features[sorted(features.columns)]

        # Split data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.15, shuffle=False)

        return x_train, x_test, y_train, y_test

    def optimize_model(self, X_train, y_train):
        """Optimize Random Forest model using GridSearchCV."""
        parameter_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 8],
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
        x_train, x_test, y_train, y_test = self.prepare_data(ticker)
        model, best_params = self.optimize_model(x_train, y_train)
        self.models[ticker] = model

        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_mse_scores = -cross_val_score(model, x_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
        cv_mae_scores = -cross_val_score(model, x_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
        cv_r2_scores = cross_val_score(model, x_train, y_train, cv=tscv, scoring='r2')

        logging.info(f'Cross-validation results for {ticker}:')
        logging.info(f'  Average MSE: {np.mean(cv_mse_scores):.4f} (+/- {np.std(cv_mse_scores) * 2:.4f})')
        logging.info(f'  Average MAE: {np.mean(cv_mae_scores):.4f} (+/- {np.std(cv_mae_scores) * 2:.4f})')
        logging.info(f'  Average R2: {np.mean(cv_r2_scores):.4f} (+/- {np.std(cv_r2_scores) * 2:.4f})')

        # Test set evaluation
        test_predictions = model.predict(x_test)
        test_mse = mean_squared_error(y_test, test_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)
        test_r2 = r2_score(y_test, test_predictions)

        logging.info(f'Test set results for {ticker}:')
        logging.info(f'  MSE: {test_mse:.4f}')
        logging.info(f'  MAE: {test_mae:.4f}')
        logging.info(f'  R2: {test_r2:.4f}')

        # Perform feature importance analysis
        # self.plot_feature_importances(model, x_train, ticker)

        return test_predictions, test_mse, test_mae, test_r2

    def plot_feature_importances(self, model, X_train, ticker):
        """Plot feature importances for the trained model."""
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = X_train.columns

        logging.info("Feature ranking:")
        for i in range(len(importances)):
            logging.info(f"{i + 1}. feature {features[indices[i]]} ({importances[indices[i]]})")

    def predict_future(self, prod_data):
        """Predict future returns for all tickers using the application data."""
        predicted_returns = {}
        tickers = self.get_tickers()
        for ticker in tickers:
            # logging.info(f'Predicting future returns for ticker: {ticker}')

            # Extract features from application data
            ticker_cols = [col for col in prod_data.columns if col.startswith(ticker)]
            economic_cols = [col for col in prod_data.columns if col.startswith('FRED_')]
            feature_cols = ticker_cols + economic_cols

            # Sort the features alphabetically
            features = prod_data[feature_cols].sort_index(axis=1)

            # Load the trained model
            model = self.models[ticker]

            # Predict the future returns
            predictions = model.predict(features)
            predicted_returns[ticker] = predictions

        # Create a DataFrame with predicted returns
        return pd.DataFrame(predicted_returns, index=prod_data.index)

    def get_tickers(self):
        """Extract tickers from the single-level columns in the DataFrame."""
        return set(col.split('_')[0] for col in self.data.columns if '_' in col and not col.startswith('FRED_'))
