from partThree import Momentum
from partOne import PartOne
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize



class PartFour:
    
    def __init__(self) -> None:
        momObj=Momentum()
        self.mom_sign=momObj.momentum_signal()
        self.returns=momObj.get_returns()
        self.alphas = None
        self.recent_returns=None
        self.portfolio_returns=None
        self.portfolio_weights=None
    
    def get_returns(self):
        file_path = r'Part 2\SMM921_pf_data_2024.xlsx'
        dataset = pd.read_excel(file_path)
        dataset['Date'] = pd.to_datetime(dataset['Date'])
        returns = dataset.drop(columns=['Date']).pct_change().dropna()
        returns['World'] = returns.mean(axis=1)
        returns.insert(0, 'Date', dataset['Date'][1:])  # Adjusting for the dropna()
        return returns

    # Alpha=IC*sigma_r*(s_i-s^-)/sigma_s
    def calculate_alphas(self, ic=0.02):
        momentum_signals=self.mom_sign[60:]
        residual_risk=self.calculate_cross_stock_average_volatility() # sigma_r
        signals = momentum_signals.drop(columns=['Date'])        
        mean_signals = signals.mean(axis=1)
        std_signals = signals.std(axis=1) # sigma_s_t
        # temp= (signals.sub(signals.iloc[:, 0:].mean(axis=1), axis=0)).div
        self.alphas = ic * residual_risk*(signals.sub(mean_signals, axis=0).div(std_signals, axis=0))
        self.alphas.insert(0, 'Date', momentum_signals['Date'])
        return f' Alphas:\n\n {self.alphas}'
    
    def calculate_cross_stock_average_volatility(self):
        recent_returns = self.returns.iloc[-60:, 1:-1]  # Most recent 60 data points
        volatilities = recent_returns.std()  # Standard deviation of each return series
        cross_stock_average_volatility = volatilities.mean()  # Cross-stock average volatility
        return cross_stock_average_volatility

    def calculate_covariance_matrix(self, window_returns):
        covariance_matrix = window_returns.cov()  # Covariance matrix of returns
        return covariance_matrix

    # Next step
    def mean_variance_optimal_weights(self, alphas, cov_matrix, risk_aversion):
        n = len(alphas)
        ones = np.ones(n)
        
        def objective(weights):
            return weights.T @ cov_matrix @ weights - risk_aversion * alphas @ weights

        constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        bounds = [(0, 1) for _ in range(n)]
        initial_guess = np.array([1/n] * n)
        result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
        if not result.success:
            raise ValueError("Optimization failed")
        return result.x

    def calculate_portfolio_returns_and_weights(self, crra=4):
        portfolio_returns = []
        portfolio_weights = []
        for i in range(60, len(self.returns) - 1):
            window_returns = self.returns.iloc[i-60:i, 1:-1]
            cov_matrix = self.calculate_covariance_matrix(window_returns)
            alphas = self.alphas.iloc[i-60, 1:]
            weights = self.mean_variance_optimal_weights(alphas, cov_matrix, crra)
            next_month_return = self.returns.iloc[i+1, 1:-1] @ (weights)
            portfolio_returns.append(next_month_return)
            portfolio_weights.append(weights)
        self.portfolio_returns = pd.Series(portfolio_returns)
        self.portfolio_weights = pd.DataFrame(portfolio_weights)
        print(f'Returns: \n\n{self.portfolio_returns}\n\nWeights: \n{self.portfolio_weights}')
        
    
    def annualisation(self):
        # Annualization factor
        annual_factor = 12
        # Annualized mean return
        mean_return_annualized = self.portfolio_returns.mean() * annual_factor
        # Annualized return volatility (standard deviation)
        return_volatility_annualized = self.portfolio_returns.std() * np.sqrt(annual_factor)
        # Sharpe ratio
        risk_free_rate = 0.02  # Assume a risk-free rate of 2%
        sharpe_ratio = (mean_return_annualized - risk_free_rate) / return_volatility_annualized
        return f'annual mean: {mean_return_annualized} annual volatility {return_volatility_annualized} sharpe ratio: {sharpe_ratio}'   

    def calculate_average_turnover(self, portfolio_weights):
        weight_changes = portfolio_weights.diff().abs().sum(axis=1)
        # Calculate average turnover
        average_turnover = weight_changes.mean()
        return average_turnover

    def plot_cumulative_returns(self):
        self.cumulative_returns=(1 + self.portfolio_returns).cumprod()
        plt.figure(figsize=(12, 8))
        self.cumulative_returns.plot(label='Cumulative Returns', marker='o')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.title('Cumulative Returns of the Optimal Portfolio')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


o = PartFour()
o.calculate_alphas()
o.calculate_portfolio_returns_and_weights()



# annualized_metrics = o.annualisation()
# print(f'Annualized Metrics: {annualized_metrics}')
# average_turnover = o.calculate_average_turnover(o.portfolio_weights)
# print(f'Average Turnover: {average_turnover}')
# o.plot_cumulative_returns()