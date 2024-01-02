import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

from src.data.FeatureEngineering import fetch_data_and_engineer_features


class ElasticNetCVStockPredictor:
    def __init__(self, df):
        self.df = df
        self.model = None

    def preprocess_data(self):
        # Convert 'Date' to datetime and extract date components
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Day'] = self.df['Date'].dt.day

        # Dropping non-numeric columns
        self.df = self.df.drop(['Ticker', 'Date'], axis=1)

        # Scale features
        scaler = StandardScaler()
        self.df.iloc[:, 1:] = scaler.fit_transform(self.df.iloc[:, 1:])

    def split_data(self):
        X = self.df.drop(['AdjClose'], axis=1)
        y = self.df['AdjClose']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
        # Set up a range of alphas and l1_ratios to try
        alphas = np.logspace(-4, 4, 50)
        l1_ratios = np.linspace(0.01, 1, 25)

        self.model = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=5, random_state=42)
        self.model.fit(X_train, y_train)

        print(f"Optimal Alpha: {self.model.alpha_}")
        print(f"Optimal l1_ratio: {self.model.l1_ratio_}")

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R-squared: {r2}")


# Usage example
processed_data = fetch_data_and_engineer_features()
predictor = ElasticNetCVStockPredictor(processed_data)
predictor.preprocess_data()
X_train, X_test, y_train, y_test = predictor.split_data()
predictor.train_model(X_train, y_train)
predictor.evaluate_model(X_test, y_test)
