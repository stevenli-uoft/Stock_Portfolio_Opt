import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

from src.data.FeatureEngineering import fetch_data_and_engineer_features


class RandomForestModel:
    def __init__(self, df):
        self.df = df
        self.model = None

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

    def train_model(self, x_train, y_train):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(x_train, y_train)

    def evaluate_model(self, x_test, y_test):
        y_pred = self.model.predict(x_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Print metrics
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R-squared: {r2}")


# Usage example
processed_data = fetch_data_and_engineer_features()
predictor = RandomForestModel(processed_data)
predictor.preprocess_data()
X_train, X_test, y_train, y_test = predictor.split_data()
predictor.train_model(X_train, y_train)
predictor.evaluate_model(X_test, y_test)
