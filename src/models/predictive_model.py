from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

from src.data.FeatureEngineering import fetch_data_and_engineer_features


class SimpleLinearModel:
    def __init__(self, df):
        self.df = df
        self.model = None

    def extract_date_components(self):
        # Convert 'Date' column to datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Extracting date components
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Day'] = self.df['Date'].dt.day

    def split_data(self):
        # Extract date components
        self.extract_date_components()

        # Assuming 'FuturePrice' is the column you want to predict
        X = self.df.drop(['Predict_AdjClose', 'Ticker', 'Date'], axis=1)  # Also dropping original 'Date' column
        y = self.df['Predict_AdjClose']

        # Split the dataset into training (80%) and testing (20%) sets
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        coef = self.model.coef_
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"COEF: {coef}")


# Usage example
processed_data = fetch_data_and_engineer_features()
processed_data['Predict_AdjClose'] = processed_data['AdjClose'].shift(-1)
processed_data.dropna(inplace=True)  # Drop NaN values after shifting
predictor = SimpleLinearModel(processed_data)
X_train, X_test, y_train, y_test = predictor.split_data()
predictor.train_model(X_train, y_train)
predictor.evaluate_model(X_test, y_test)
