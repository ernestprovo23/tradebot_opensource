import numpy as np
from scipy.optimize import minimize

def optimize_portfolio(expected_returns, covariance_matrix, risk_aversion, total_investment):
    def objective(weights):
        portfolio_return = np.dot(expected_returns, weights)
        portfolio_risk = np.dot(weights.T, np.dot(covariance_matrix, weights))
        return -(portfolio_return - risk_aversion * portfolio_risk)

    # Constraints (weights must sum to 1)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # Bounds for weights
    bounds = [(0, 1) for asset in range(len(expected_returns))]

    # Initial guess (equal weighting)
    initial_weights = [1./len(expected_returns) for asset in expected_returns]

    # Run optimization
    solution = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights = solution.x

    # Calculate quantities to purchase for each asset
    quantities_to_purchase = optimal_weights * total_investment / expected_returns

    return quantities_to_purchase