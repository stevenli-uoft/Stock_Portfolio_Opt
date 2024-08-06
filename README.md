# Enhanced Mean-Variance Optimization with Machine Learning

## Project Overview
### Abstract
This project explores an innovative approach to portfolio optimization by enhancing the traditional Mean-Variance Optimization (MVO) model with Machine Learning techniques. Specifically, I employed a Random Forest Regression (RFR) to predict stock returns, aiming to improve upon the standard MVO model's performance in portfolio allocation. My project demonstrates how integrating machine learning with established financial models can potentially yield better risk-adjusted returns, particularly for lower-risk investment strategies.
### Mean-Variance Optimization (MVO) Concept
Mean-Variance Optimization, introduced by Harry Markowitz in 1952, is a foundational concept in modern portfolio theory. It aims to construct an optimal portfolio by balancing the trade-off between expected returns (mean) and risk (variance). The traditional MVO model uses historical data to estimate expected returns and covariances between assets, then determines the asset allocation that maximizes returns for a given level of risk or minimizes risk for a given level of return.

My project enhances this classical approach by using machine learning to generate quarterly (3-month) forward-looking return estimates instead of relying solely on historical averages. This modification aims to capture more complex patterns in financial data and potentially improve the model's adaptability to changing market conditions.

For example, a typical MVO model will use the mean of Q1 2024 stock returns as expected returns for Q2 2024 for its portfolio optimization. Introducing an ML model will predict the expected returns of Q2 2024, instead of using the mean of the previous quarter. This forward-looking approach could provide investors and financial institutions with a more dynamic and responsive tool for portfolio management, potentially leading to improved investment outcomes in various market conditions.

## Key Features
- Extensive data collection and preprocessing from Yahoo Finance (stock data) and FRED (economic data)
- Advanced feature engineering to create relevant predictors for weekly stock returns, including technical indicators and economic factors
- Implementation of Random Forest Regression for 3-month weekly return prediction
- Rigorous hyperparameter tuning using GridSearchCV with time series cross-validation
- Comprehensive comparison of ML-enhanced MVO with traditional "baseline" MVO
- Extensive backtesting across multiple training and testing date ranges (2022 Q3 to 2024 Q2), risk levels, and portfolio structures
- In-depth performance evaluation using various metrics including Sharpe ratio, Mean Squared Error (MSE), and R-squared

## Technologies Used
- Python 3.8+
- pandas and numpy for efficient data manipulation and numerical operations
- scikit-learn for machine learning models and preprocessing
- scipy for optimization algorithms
- yfinance for fetching stock data
- fredapi for accessing economic indicators from Federal Reserve Economic Data

## Methodology Highlights
- **Data Preprocessing**: Implemented effective data cleaning techniques to handle missing values and outliers, ensuring good quality input for the models. This process focused on maintaining the integrity of the time series data.
- **Feature Engineering**: Developed over 20 engineered features, including moving averages, volatility measures, and economic indicators. These features were designed to capture important market trends and improve the Random Forest Regression (RFR) model's performance.
- **Feature Importance Analysis**: Analyzed which features were most important for the model's predictions. This helped in deciding which features to keep, remove, or improve.
- **Model Tuning**: Spent significant time testing and improving the RFR model. Used Grid Search Cross-Validation to find the best combination of model parameters and applied regularization techniques to prevent overfitting.
- **Time Series Validation**: Used TimeSeriesSplit for model validation, which helped ensure the model was tested realistically on data it hadn't seen before, similar to how it would be used in real-world scenarios.

## Results and Findings
The ML-enhanced MVO model demonstrated several advantages over the traditional MVO approach, as evidenced by the graph:

- **Higher Average Sharpe Ratios**: The blue line representing the ML model's Sharpe ratio is consistently above the green line of the baseline model across all volatility levels. This indicates that the ML model achieved better risk-adjusted returns, particularly at lower risk levels (0.05-0.25 volatility).
- **Better Predictive Accuracy**: The dark blue bars (ML Predicted Returns) are generally closer to the light blue bars (ML Actual Returns) compared to the dark green (Baseline Predicted Returns) and light green (Baseline Actual Returns) bars. This suggests that the ML model's predictions were more accurate, especially at lower volatility levels.
- **Adaptability to Market Conditions**: The consistency of the ML model's performance across different volatility levels, as shown by the relatively stable height of the blue bars, suggests better adaptability to varying market conditions.
- **Improved Performance at Lower Risk Levels**: The graph clearly shows that the ML model outperforms the baseline particularly well at lower volatility levels (0.05-0.25). The difference in bar heights between ML and baseline is most pronounced in this range, indicating the ML model's suitability for conservative investment strategies.
- **Higher Actual Returns**: The light blue bars (ML Actual) are generally higher than the light green bars (Baseline Actual), especially at lower volatility levels. This indicates that the ML model's portfolios achieved higher actual returns in most cases.

However, it's important to note some limitations:

- **Overestimation of Returns**: Both models show a tendency to overestimate returns, as evidenced by the predicted bars (dark blue and dark green) often being higher than the actual return bars (light blue and light green).
- **Diminishing Advantage at Higher Risk Levels**: The performance gap between ML and baseline models narrows at higher volatility levels (0.35-0.45), with the baseline occasionally outperforming in terms of actual returns.
- **Sharpe Ratio Decline**: Both models show declining Sharpe ratios as volatility increases, which is expected, but the ML model maintains a higher Sharpe ratio throughout.

The graph visualizes results averaged across 8 quarters (Q3 2022 to Q2 2024), with 5 levels of risk tested for each quarter. This comprehensive testing demonstrates the ML model's consistent outperformance, particularly in lower-risk scenarios.

![test_result_plot](https://github.com/user-attachments/assets/aa51b7a1-c8b1-4c74-9316-ce83163f2ef6)

These results suggest that the ML-enhanced MVO model offers significant improvements in portfolio optimization, especially for lower-risk strategies. The model's ability to more accurately predict returns and achieve higher Sharpe ratios could be particularly valuable for risk-averse investors or in volatile market conditions. However, the diminishing advantage at higher risk levels indicates that further refinement could potentially enhance the model's performance across all risk levels.

## Future Improvements
- **Explore Advanced Techniques**: Look into using more advanced methods like XGBoost, which might improve on the current Random Forest model's performance.
- **Enhance Feature Set**: Continue improving the model by creating new features and incorporating more economic data to better capture market behavior.
- **Try Other ML Models**: Experiment with different types of machine learning models to see if they can predict stock returns more accurately.
- **Improve Risk Analysis**: Develop better ways to measure and manage risk in the portfolio, possibly by including more advanced risk metrics in the analysis.


## Project Structure
```
├── src/
│   ├── data/
│   │   ├── DataCollection.py
│   │   └── DataHandling.py
│   ├── models/
│   │   └── MVOModel.py
│   │   └── RFRModel.py
│   ├── tests/
│   │   └── InvestmentSimulator.py
│   │   └── diverse_portfolio_sample
│   └── visualization/
│       └── test_result_plot.png
│       └── visualizer.py
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

## Contact Information
Steven Li

https://www.linkedin.com/in/steven-li-uoft/

Feel free to reach out if you have any questions or would like to discuss this project further!
