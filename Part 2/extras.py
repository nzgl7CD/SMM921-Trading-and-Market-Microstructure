from partThree import Momentum
from partOne import PartOne
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from exogenous import Exogenous as exo



class PartFour:
    
    def __init__(self) -> None:
        momObj=Momentum()
        self.mom_sign=momObj.momentum_signal()
        self.returns=momObj.get_returns()
        self.alphas = None
        self.recent_returns=None
        self.portfolio_returns=None
        self.portfolio_weights=None
        self.risk_free=exo.
        
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
        momentum_signals = self.mom_sign[59:]
        residual_risk = self.calculate_cross_stock_average_volatility()  # sigma_r
        signals = momentum_signals.drop(columns=['Date'])
        mean_signals = signals.mean(axis=1)
        std_signals = signals.std(axis=1)  # sigma_s_t
        self.alphas = ic * residual_risk * (signals.sub(mean_signals, axis=0).div(std_signals, axis=0))
        self.alphas.insert(0, 'Date', momentum_signals['Date'])
        return f'Alphas:\n\n{self.alphas}'

    def calculate_cross_stock_average_volatility(self):
        recent_returns = self.returns.iloc[-60:, 1:-1]  # Most recent 60 data points
        volatilities = recent_returns.std()  # Standard deviation of each return series
        cross_stock_average_volatility = volatilities.mean()  # Cross-stock average volatility
        return cross_stock_average_volatility

    def calculate_covariance_matrix(self, window_returns):
      # Calculate the sample covariance matrix
      covariance_matrix = window_returns.cov()

      # Replace all cross-asset correlations with the average cross-asset correlation
      avg_correlation = (covariance_matrix.values - np.diag(np.diag(covariance_matrix.values))).mean()
      adjusted_covariance_matrix = avg_correlation + np.diag(np.diag(covariance_matrix.values))

      return pd.DataFrame(adjusted_covariance_matrix, index=covariance_matrix.index, columns=covariance_matrix.columns)

    # Next step
    def mean_variance_optimal_weights(self, alphas, cov_matrix, risk_aversion):
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        ones = np.ones(len(alphas))

        # Calculate theta_t
        numerator = ones.T @ inv_cov_matrix @ alphas
        denominator = ones.T @ inv_cov_matrix @ ones
        theta = numerator / denominator

        # Calculate optimal weights
        weights = (1 / risk_aversion) * inv_cov_matrix @ (alphas - ones * theta)
        return weights

    def calculate_portfolio_returns_and_weights(self, crra=4):
        portfolio_returns = []
        portfolio_weights = []
        for i in range(60, len(self.returns) - 1):
            window_returns = self.returns.iloc[i-60:i, 1:-1]
            cov_matrix = self.calculate_covariance_matrix(window_returns)
            alphas = self.alphas.iloc[i-60].drop('Date').values
            weights = self.mean_variance_optimal_weights(alphas, cov_matrix, crra)
            next_month_return = self.returns.iloc[i+1, 1:-1].dot(weights)
            portfolio_returns.append(next_month_return)
            portfolio_weights.append(weights)
        self.portfolio_returns = pd.Series(portfolio_returns)
        self.portfolio_weights = pd.DataFrame(portfolio_weights)
        print(f'Returns: \n\n{self.portfolio_returns}\n\nWeights: \n{self.portfolio_weights}')
        return self.portfolio_weights
        
    
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
        optimal_metric = pd.DataFrame({
            'Mean Return': [mean_return_annualized],
            'Standard Deviation': [return_volatility_annualized],
            'Sharpe Ratio': [sharpe_ratio]
        }, index=['Optimal'])

        portfolio_metrics = portfolio_metrics = pd.concat([optimal_metric])

        return pd.DataFrame({
            'Mean Return': [mean_return_annualized],
            'Standard Deviation': [return_volatility_annualized],
            'Sharpe Ratio': [sharpe_ratio]
        }, index=['Optimal'])   

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
    

    def investigate_portfolio_effects(self, ic_range, risk_aversion_range):
        results = []
        for ic in ic_range:
            for risk_aversion in risk_aversion_range:
                self.calculate_alphas(ic)
                self.calculate_portfolio_returns_and_weights(risk_aversion)
                annualized_metrics = self.annualisation()
                average_turnover = self.calculate_average_turnover(self.portfolio_weights)
                results.append({
                    'IC': ic,
                    'Risk Aversion': risk_aversion,
                    'Mean Return (Annualized)': annualized_metrics['Mean Return'].iloc[0],
                    'Standard Deviation (Annualized)': annualized_metrics['Standard Deviation'].iloc[0],
                    'Sharpe Ratio': annualized_metrics['Sharpe Ratio'].iloc[0],
                    'Average Turnover': average_turnover
                })
        
        return pd.DataFrame(results)

    

        


o = PartFour()
o.calculate_alphas()
o.calculate_portfolio_returns_and_weights()
o.annualisation()
w=o.calculate_portfolio_returns_and_weights()
o.calculate_average_turnover(w)
o.plot_cumulative_returns()



ic_range = [0.01, 0.02, 0.03]
risk_aversion_range = [3, 4, 5]

# Investigate portfolio effects
# results_df = o.investigate_portfolio_effects(ic_range, risk_aversion_range)

# print(results_df)

