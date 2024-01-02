import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

from src.data.FeatureEngineering import fetch_data_and_engineer_features


class RandomForestTuner:
    def __init__(self, df):
        self.df = df
        self.model = RandomForestRegressor(random_state=42)

    def preprocess_data(self):
        # Convert 'Date' to datetime and extract date components
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Day'] = self.df['Date'].dt.day

        # Dropping non-numeric columns (if not used as features)
        self.df = self.df.drop(['Ticker', 'Date'], axis=1)

    def split_data(self):
        # Assuming 'AdjClose' is the target variable
        x = self.df.drop(['AdjClose'], axis=1)
        y = self.df['AdjClose']

        # Split the dataset into training (80%) and testing (20%) sets
        return train_test_split(x, y, test_size=0.2, random_state=42)

    def tune_hyperparameters(self, X_train, y_train):
        # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid,
                                   cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

        grid_search.fit(X_train, y_train)
        print("Best Parameters:", grid_search.best_params_)

        # Update the model with the best parameters
        self.model = grid_search.best_estimator_

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"MSE: {mse}")
        print(f"R-squared: {r2}")


# Usage example
processed_data = fetch_data_and_engineer_features()
tuner = RandomForestTuner(processed_data)
tuner.preprocess_data()
X_train, X_test, y_train, y_test = tuner.split_data()
tuner.tune_hyperparameters(X_train, y_train)
tuner.evaluate_model(X_test, y_test)

# Best Parameters: {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 200}
# MSE: 4.066279322141551
# R-squared: 0.9989358577063313
