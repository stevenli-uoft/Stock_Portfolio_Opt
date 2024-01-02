import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

from src.data.FeatureEngineering import fetch_data_and_engineer_features


class RandomForestCrossValidator:
    def __init__(self, df, n_estimators, max_depth, min_samples_split):
        self.df = df
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def preprocess_data(self):
        # Convert 'Date' to datetime and extract date components
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Day'] = self.df['Date'].dt.day

        # Dropping non-numeric columns (if not used as features)
        self.df = self.df.drop(['Ticker', 'Date'], axis=1)

    def perform_cross_validation(self):
        X = self.df.drop(['AdjClose'], axis=1)
        y = self.df['AdjClose']

        # Initialize Random Forest with the tuned parameters
        model = RandomForestRegressor(n_estimators=self.n_estimators,
                                      max_depth=self.max_depth,
                                      min_samples_split=self.min_samples_split,
                                      random_state=42)

        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)
        return rmse_scores

    def display_scores(self, scores):
        print("Scores:", scores)
        print("Mean:", scores.mean())
        print("Standard deviation:", scores.std())


# Usage example
processed_data = fetch_data_and_engineer_features()
validator = RandomForestCrossValidator(processed_data, n_estimators=200, max_depth=10, min_samples_split=2)
validator.preprocess_data()
scores = validator.perform_cross_validation()
validator.display_scores(scores)
