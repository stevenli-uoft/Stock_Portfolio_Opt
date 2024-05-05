import pandas as pd
import numpy as np
from scipy.optimize import minimize


class PortfolioOptimizer:
    def __init__(self, predicted_prices_df, max_volatility=None):
        self.prices = predicted_prices_df
        self.returns = self.calculate_returns()
        self.cov_matrix = self.returns.cov()
        self.expected_returns = self.returns.mean()
        self.max_volatility = max_volatility  # Maximum volatility the user is willing to accept

    def calculate_returns(self):
        """Calculate daily returns of stocks."""
        return self.prices.pct_change().dropna()

    def objective_function(self, weights):
        """Objective function to minimize for maximizing return under given risk."""
        portfolio_return = np.dot(weights, self.expected_returns)
        return -portfolio_return  # Negate since we want to maximize

    def constraint_volatility(self, weights):
        """Constraint to ensure portfolio volatility does not exceed max_volatility."""
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return self.max_volatility - portfolio_volatility

    def optimize_portfolio(self):
        """Find the optimal weights to maximize return under specified risk preferences."""
        num_assets = len(self.prices.columns)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights must sum to 1
        if self.max_volatility:
            constraints.append({'type': 'ineq', 'fun': self.constraint_volatility})  # Volatility must be under limit
        bounds = tuple((0, 1) for asset in range(num_assets))  # No short selling
        initial_guess = np.array([1. / num_assets] * num_assets)  # Equal distribution
        result = minimize(self.objective_function, initial_guess, method='SLSQP', bounds=bounds,
                          constraints=constraints)
        return result

    def get_optimal_weights(self):
        """Get the optimal portfolio weights."""
        optimal_result = self.optimize_portfolio()
        if optimal_result.success:
            # Map weights to ticker names for clear output
            weight_dict = dict(zip(self.prices.columns, optimal_result.x))
            return weight_dict
        else:
            raise Exception("Optimization did not converge")
