from partThree import Momentum
from partOne import PartOne
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt



class PartFour:
    
    def __init__(self) -> None:
        momObj=Momentum()
        self.mom_sign=momObj.momentum_signal()
        self.returns=momObj.get_returns()
        self.alphas = None
        self.recent_returns=None
        self.portfolio_returns=None
        self.portfolio_weights=None

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

    def calculate_covariance_matrix(self):
        recent_returns = self.returns.iloc[-60:, 1:]  # Most recent 60 data points
        covariance_matrix = recent_returns.cov()  # Covariance matrix of returns
        return covariance_matrix

    # Next step
    def mean_variance_optimal_weights(self, alpha, sigma, crra):
        # inv_cov_matrix = np.linalg.inv(cov_matrix)
        myOnes = np.ones(len(alpha.index))
        inv_sigma = np.linalg.inv(sigma)
        # weights = inv_cov_matrix.dot(mean_returns - risk_aversion * cov_matrix.dot(ones)) / (ones.dot(inv_cov_matrix).dot(mean_returns - risk_aversion * cov_matrix.dot(ones)))
        # return weights
        gamma = (alpha.T @ inv_sigma @ myOnes - crra)/(myOnes.T @ inv_sigma @ myOnes)
        w = inv_sigma @ (alpha - myOnes*gamma) / crra
        print(sum(w))
        return w

    def calculate_portfolio_returns_and_weights(self,crra=4):
        
        # portfolio_returns = []
        # portfolio_weights = []
        # gamma=None
        # w=None
        # for i in range(61,len(self.returns.drop(columns=['Date']))): 
        #     alpha = self.returns.drop(columns=['Date']).loc[i:].dropna().mean()
        #     sigma = self.returns.drop(columns=['Date']).loc[i:].dropna().cov()
        #     inv_sigma = np.linalg.inv(sigma)
        #     myOnes = np.ones(35)

        #     gamma = (alpha.T @ inv_sigma @ myOnes - crra)/(myOnes.T @ inv_sigma @ myOnes)
        #     w = inv_sigma @ (alpha - myOnes*gamma) / crra
        # print("Sum of absolute weights is " + str(sum(abs(w))))
        # return f'Weights: \n{w}'

        
        # rolling_windows = returns_without_date.rolling(window=60)
        portfolio_returns = []
        portfolio_weights = []
        returns_without_date=self.returns.drop(columns=['Date', 'World']).dropna()

        for i in range(61, len(returns_without_date)):
            window_returns = returns_without_date.loc[i:]
            
            mean_returns = window_returns.mean()
            cov_matrix = window_returns.cov()
            # First month return 0.06021854209088286
            weights = self.mean_variance_optimal_weights(mean_returns, cov_matrix, crra)
            next_month=i+1
            if next_month < returns_without_date.index.size:
                temp=window_returns.loc[i+1]
                next_month_return = temp.dot(weights)
                portfolio_returns.append(next_month_return)
                portfolio_weights.append(weights)

        self.portfolio_returns = pd.Series(portfolio_returns)
        self.portfolio_weights = pd.DataFrame(portfolio_weights)
        # returns_str = portfolio_returns.to_string(header=True, index=True, float_format='{:,.4f}'.format)
        print(f'Returns: \n\n{self.cumulative_returns}\n\nWeights: \n{self.portfolio_weights}')
        return f'Returns: \n\n{self.portfolio_returns}\n\nWeights: \n{self.portfolio_weights}'
    

    def calculate_metrics(self):
        # Annualization factor
        annual_factor = 12

        # Annualized mean return
        mean_return_annualized = self.portfolio_returns.mean() * annual_factor

        # Annualized return volatility (standard deviation)
        return_volatility_annualized = self.portfolio_returns.std() * np.sqrt(annual_factor)

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

    def plot_cumulative_returns(self):
        

        plt.figure(figsize=(12, 8))
        self.cumulative_returns.plot(label='Cumulative Returns', marker='o')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.title('Cumulative Returns of the Optimal Portfolio')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
        







o=PartFour()
o.calculate_alphas()
o.calculate_covariance_matrix()
o.calculate_portfolio_returns_and_weights()
# o.calculate_metrics()
# o.plot_cumulative_returns()