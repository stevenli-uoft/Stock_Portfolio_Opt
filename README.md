# Enhanced Mean-Variance Optimization with Machine Learning

## Project Overview
This project demonstrates an innovative approach to portfolio optimization by enhancing the traditional Mean-Variance Optimization (MVO) model with Machine Learning techniques. By leveraging Random Forest Regression (RFR) to predict stock returns, this project aims to improve upon the standard MVO model's performance in portfolio allocation.

## Key Features
- Extensive data collection and preprocessing from Yahoo Finance (stock data) and FRED (economic data)
- Advanced feature engineering to create relevant predictors for weekly stock returns, including technical indicators and economic factors
- Implementation of Random Forest Regression for 3-month weekly return prediction
- Rigorous hyperparameter tuning using GridSearchCV with time series cross-validation
- Comprehensive comparison of ML-enhanced MVO with traditional "baseline" MVO
- Extensive backtesting across multiple date ranges (2022 Q3 to 2024 Q2), risk levels, and portfolio structures
- In-depth performance evaluation using various metrics including Sharpe ratio, Mean Squared Error (MSE), and R-squared

## Technologies Used
- Python 3.8+
- pandas and numpy for efficient data manipulation and numerical operations
- scikit-learn for machine learning models and preprocessing
- scipy for optimization algorithms
- yfinance for fetching stock data
- fredapi for accessing economic indicators from Federal Reserve Economic Data


## Project Structure
```
├── src/
│   ├── data/
│   │   ├── DataCollection.py
│   │   └── DataHandling.py
│   ├── models/
│   │   └── StockPricePredictor.py
│   ├── optimization/
│   │   └── MVOModel.py
│   └── tests/
│       └── InvestmentSimulator.py
│       └── sample_data
├── main.py
├── requirements.txt
└── README.md
```

## Installation and Usage

1. Clone this repository
2. Install the required packages:
```
pip install -r requirements.txt
```
3. Input your FRED API KEY. To get a key, apply through this link: https://fred.stlouisfed.org/docs/api/fred/
4. Run the main script
```
python main.py
```
5. Adjust variables such as TRAINING_START, TRAINING_END, date_ranges, risk_pref_test, and the stock portfolio in FILE_PATH to play around with the model

## Methodology Highlights
- **Data Preprocessing**: Implemented effective data cleaning techniques to handle missing values and outliers, ensuring good quality input for the models. This process focused on maintaining the integrity of the time series data.
- **Feature Engineering**: Developed over 20 engineered features, including moving averages, volatility measures, and economic indicators. These features were designed to capture important market trends and improve the Random Forest Regression (RFR) model's performance.
- **Feature Importance Analysis**: Analyzed which features were most important for the model's predictions. This helped in deciding which features to keep, remove, or improve.
- **Model Tuning**: Spent significant time testing and improving the RFR model. Used Grid Search Cross-Validation to find the best combination of model parameters and applied regularization techniques to prevent overfitting.
- **Time Series Validation**: Used TimeSeriesSplit for model validation, which helped ensure the model was tested realistically on data it hadn't seen before, similar to how it would be used in real-world scenarios.

## Results and Findings
After reviewing the data and the visualization, I'll rewrite the Results and Findings section for your README. This version will reflect the performance of your ML-enhanced MVO model compared to the traditional baseline MVO approach:

## Results and Findings
The ML-enhanced MVO model demonstrated several advantages over the traditional MVO approach:
1. **Higher Average Sharpe Ratios**: The ML model consistently achieved higher Sharpe ratios across different volatility levels, indicating better risk-adjusted returns. This is particularly evident at lower risk levels (0.05-0.25 volatility).
2. **Better Predictive Accuracy**: On average, the ML model's predicted returns were closer to actual returns compared to the baseline model, especially at lower volatility levels.
3. **Adaptability to Market Conditions**: The ML model showed more consistent performance across different quarters, suggesting better adaptability to changing market environments.
4. **Improved Performance at Lower Risk Levels**: The ML model outperformed the baseline particularly well at lower volatility levels (0.05-0.25), making it potentially more suitable for conservative investment strategies.
5. **Resilience in Volatile Periods**: During quarters with negative returns (e.g., 2022 Q3), the ML model generally predicted and achieved less negative returns than the baseline model.

However, it's important to note:
- Both models sometimes overestimated returns, especially in down markets.
- The baseline model occasionally outperformed in terms of actual returns, particularly at higher risk levels.
- The ML model's performance advantage was less pronounced at very high risk levels (0.45 volatility).

[Insert visualization here]

These results suggest that the ML-enhanced MVO model offers improvements in portfolio optimization, particularly for lower-risk strategies. However, further refinement could potentially enhance its performance across all risk levels.
  
The output from main.py spanned 8 quarters, with 4 levels of risk for each quarter. This was the final and main test of my project.

## Future Improvements
- **Explore Advanced Techniques**: Look into using more advanced methods like XGBoost, which might improve on the current Random Forest model's performance.
- **Enhance Feature Set**: Continue improving the model by creating new features and incorporating more economic data to better capture market behavior.
- **Try Other ML Models**: Experiment with different types of machine learning models to see if they can predict stock returns more accurately.
- **Improve Risk Analysis**: Develop better ways to measure and manage risk in the portfolio, possibly by including more advanced risk metrics in the analysis.

## Contact Information
Steven Li

https://www.linkedin.com/in/steven-li-uoft/

Feel free to reach out if you have any questions or would like to discuss this project further!
