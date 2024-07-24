import numpy as np
import pandas as pd
from scipy.optimize import minimize


class PortfolioOptimizer:
    def __init__(self, returns_df, target_volatility, risk_free_rate=0.02):
        self.returns = returns_df
        self.target_volatility = target_volatility
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(returns_df.columns)
        self.asset_names = returns_df.columns

    def _calculate_portfolio_stats(self, weights):
        portfolio_return = np.sum(self.returns.mean() * weights)  # 3-month return
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(self.returns.cov() * 52, weights)))  # Annualized volatility
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def _objective_function(self, weights):
        return -self._calculate_portfolio_stats(weights)[2]  # Negative Sharpe ratio

    def _volatility_constraint(self, weights):
        return self.target_volatility - self._calculate_portfolio_stats(weights)[1]

    def get_optimal_weights(self):
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            {'type': 'eq', 'fun': self._volatility_constraint}  # Target volatility constraint
        )
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_weights = np.array([1.0 / self.num_assets] * self.num_assets)

        result = minimize(
            self._objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        optimal_return, optimal_volatility, optimal_sharpe = self._calculate_portfolio_stats(optimal_weights)

        print(f"Optimizing portfolio for {self.target_volatility} volatility:")
        return {
            'Weights': dict(zip(self.asset_names, optimal_weights)),
            'Return': optimal_return,
            'Volatility': optimal_volatility,
            'Sharpe Ratio': optimal_sharpe
        }
