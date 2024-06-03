from partThree import Momentum
from partTwo import PortfolioAnalysis
from partOne import PartOne
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt



class PartFour:
    
    def __init__(self, filepath) -> None:
        momentumClass=Momentum(filepath)
        self.momentum_signals=momentumClass.momentum_signal()
        self.alphas = None
        o1=PartOne
        self.returns=o1.get_returns()
        self.cross_stock_average_volatility = self.calculate_cross_stock_average_volatility()
        self.covariance_matrix = self.calculate_covariance_matrix()
    
    def calculate_cross_stock_average_volatility(self):
        recent_returns = self.returns.iloc[-60:]  # Most recent 60 data points
        volatilities = recent_returns.std()  # Standard deviation of each return series
        cross_stock_average_volatility = volatilities.mean()  # Cross-stock average volatility
        return cross_stock_average_volatility

    def calculate_covariance_matrix(self):
        recent_returns = self.returns.iloc[-60:]  # Most recent 60 data points
        covariance_matrix = recent_returns.cov()  # Covariance matrix of returns
        return covariance_matrix

    def mean_variance_optimal_weights(self, mean_returns, cov_matrix, risk_aversion):
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        ones = np.ones(len(mean_returns))
        weights = inv_cov_matrix.dot(mean_returns - risk_aversion * cov_matrix.dot(ones)) / (ones.dot(inv_cov_matrix).dot(mean_returns - risk_aversion * cov_matrix.dot(ones)))
        return weights

    def calculate_portfolio_returns_and_weights(self, risk_aversion):
        rolling_windows = self.returns.rolling(window=60)
        portfolio_returns = []
        portfolio_weights = []

        for i in range(60, len(self.returns)):
            window_returns = self.returns.iloc[i-60:i]
            mean_returns = window_returns.mean()
            cov_matrix = window_returns.cov()
            
            weights = self.mean_variance_optimal_weights(mean_returns, cov_matrix, risk_aversion)
            next_month_return = self.returns.iloc[i].dot(weights)
            portfolio_returns.append(next_month_return)
            portfolio_weights.append(weights)

        portfolio_returns = pd.Series(portfolio_returns, index=self.returns.index[60:])
        portfolio_weights = pd.DataFrame(portfolio_weights, index=self.returns.index[60:], columns=self.returns.columns)
        return portfolio_returns, portfolio_weights

    def calculate_metrics(self, portfolio_returns):
        # Annualization factor
        annual_factor = 12

        # Annualized mean return
        mean_return_annualized = portfolio_returns.mean() * annual_factor

        # Annualized return volatility (standard deviation)
        return_volatility_annualized = portfolio_returns.std() * np.sqrt(annual_factor)

        # Sharpe ratio
        risk_free_rate = 0.02  # Assume a risk-free rate of 2%
        sharpe_ratio = (mean_return_annualized - risk_free_rate) / return_volatility_annualized

        return mean_return_annualized, return_volatility_annualized, sharpe_ratio

    def calculate_average_turnover(self, portfolio_weights):
        # Calculate the absolute change in weights from one month to the next
        weight_changes = portfolio_weights.diff().abs().sum(axis=1)
        # Calculate average turnover
        average_turnover = weight_changes.mean()
        return average_turnover

    def plot_cumulative_returns(self, portfolio_returns):
        cumulative_returns = (1 + portfolio_returns).cumprod() - 1

        plt.figure(figsize=(12, 8))
        cumulative_returns.plot(label='Cumulative Returns', marker='o')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.title('Cumulative Returns of the Optimal Portfolio')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
        







output_path = r'Part 2\modified_SMM921_pf_data_2024.xlsx'
o=PartFour(output_path)
o.calculate_cross_stock_average_volatility()
o.calculate_covariance_matrix()

risk_aversion_coefficient = 4
portfolio_returns, portfolio_weights = o.calculate_portfolio_returns_and_weights(risk_aversion_coefficient)

# Calculate the metrics for the portfolio
mean_return_annualized, return_volatility_annualized, sharpe_ratio = o.calculate_metrics(portfolio_returns)
print(mean_return_annualized, return_volatility_annualized, sharpe_ratio)

# Calculate average monthly turnover
average_turnover = o.calculate_average_turnover(portfolio_weights)
print(average_turnover)

# Plot the cumulative returns
o.plot_cumulative_returns(portfolio_returns)

