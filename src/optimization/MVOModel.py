import numpy as np
from scipy.optimize import minimize


class PortfolioOptimizer:
    def __init__(self, returns_df, max_volatility, risk_free_rate=0.02):
        self.returns = returns_df
        self.max_volatility = max_volatility
        self.risk_free_rate = risk_free_rate  # Assuming this is an annual rate
        self.n_assets = len(returns_df.columns)

        # Assuming returns_df contains 6-month return predictions
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()

    def _portfolio_return(self, weights):
        return np.sum(self.mean_returns * weights)  # Already 6-month return

    def _portfolio_volatility(self, weights):
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))  # 6-month volatility

    def _sharpe_ratio(self, weights):
        # Adjust risk-free rate to 6-month period
        rf_6month = (1 + self.risk_free_rate) ** 0.5 - 1
        return (self._portfolio_return(weights) - rf_6month) / self._portfolio_volatility(weights)

    def _negative_sharpe_ratio(self, weights):
        return -self._sharpe_ratio(weights)

    def _constraint_sum(self, weights):
        return np.sum(weights) - 1

    def _constraint_volatility(self, weights):
        return self.max_volatility - self._portfolio_volatility(weights)

    def get_optimal_weights(self):
        constraints = (
            {'type': 'eq', 'fun': self._constraint_sum},
            {'type': 'ineq', 'fun': self._constraint_volatility}
        )
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.array([1 / self.n_assets] * self.n_assets)

        result = minimize(
            self._negative_sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        portfolio_return = self._portfolio_return(optimal_weights)
        portfolio_volatility = self._portfolio_volatility(optimal_weights)
        sharpe_ratio = self._sharpe_ratio(optimal_weights)

        # Annualize the 6-month return and volatility for reporting
        annualized_return = (1 + portfolio_return) ** 2 - 1
        annualized_volatility = portfolio_volatility * np.sqrt(2)

        print(f"MVO at {self.max_volatility} volatility preference")
        return {
            'Weights': dict(zip(self.returns.columns, optimal_weights)),
            'Return': annualized_return,
            'Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio
        }
