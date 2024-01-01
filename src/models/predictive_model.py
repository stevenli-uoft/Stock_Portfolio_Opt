from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

from src.data.FeatureEngineering import FeatureEngineeringHandler


class StockPricePredictor:
    def __init__(self, df):
        self.df = df
        self.model = None

    def split_data(self):
        # Assuming 'FuturePrice' is the column you want to predict
        X = processed_data.drop(['Next Day Close', 'Ticker'], axis=1)
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
        print(f"RMSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"RMSE: {coef}")


# Usage example
handler = FeatureEngineeringHandler()
processed_data = handler.fetch_data_and_engineer_features()
processed_data['Predict_AdjClose'] = processed_data['AdjClose'].shift(-1)
predictor = StockPricePredictor(processed_data)
X_train, X_test, y_train, y_test = predictor.split_data()
predictor.train_model(X_train, y_train)
predictor.evaluate_model(X_test, y_test)
