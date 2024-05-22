# Stock Portfolio Optimization Project

## Overview

This project was developed as part of my personal endeavor to enhance my skills in data science and finance by building a robust model from scratch and learning from my mistakes along the way. This project aims to optimize a users stock portfolio by predicting stock prices using a Random Forest Regression model and determining the optimal allocation of stocks based on predicted prices using Mean-Variance Optimization (MVO). The goal is to achieve a specific risk-return ratio tailored to the user's risk preference and their existing stock portfolio.

## Methodology

### Data Collection

1. **Stock Data**: Historical stock data was collected from Yahoo Finance using the `yfinance` library. Stock data is collected based on the tickers that are in the users inputted stock portfolio.
2. **Economic Data**: Relevant economic indicators were collected from the Federal Reserve Economic Data (FRED) using the `fredapi` library.

### Data Handling

1.  **Data Handling**: Stock Dataframe and Economic Dataframe were joined together by date index.
2.  **Data Cleaning**: Missing economic data values were handled using forward and backward filling. Rows with remaining missing values in stock data were dropped.
3. **Feature Engineering**: Features such as moving averages (short and long-term), exponential moving averages, volatility (short and long-term), daily returns, and momentum features were added to enhance the model's predictive accuracy.

### Model Building

1. **Random Forest Regression**:
   - **Hyperparameter Tuning**: GridSearchCV was used to find the optimal hyperparameters.
   - **Feature Scaling**: All features were scaled using `StandardScaler` to standardize the data.
   - **Regularization**: Regularization techniques were applied to prevent overfitting.

2. **Mean-Variance Optimization (MVO)**:
   - The predicted stock prices were used to determine the optimal portfolio allocation using the `scipy.optimize.minimize` function.

### Performance Evaluation

1. **Cross-Validation**: The model was evaluated using TimeSeriesSplit cross-validation.
2. **Final Testing**: The model's performance was validated with new portfolios and datasets to ensure robustness.

## Libraries Used

- `pandas`
- `numpy`
- `yfinance`
- `fredapi`
- `scikit-learn`
- `scipy`
- `matplotlib`

## Key Findings

### Feature Importance Analysis

The following features were found to have the biggest influence on stock prediction:

### Results from Final Testing

The final model was tested on new portfolios and datasets, yielding the following results:

## Conclusion

This project successfully demonstrated the use of machine learning and optimization techniques to create a robust stock portfolio. The inclusion of economic indicators significantly improved the model's accuracy, iterative improvements through hyperparamter tuning helped reduce overfitting, and further feature engineer allowed for building a robust model.

