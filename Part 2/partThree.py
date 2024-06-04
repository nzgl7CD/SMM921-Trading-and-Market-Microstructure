from partTwo import PortfolioAnalysis
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from partOne import PartOne
from statsmodels.formula.api import ols

class Momentum:
    def __init__(self) -> None:
        o = PortfolioAnalysis()
        self.dataset = o.get_data()
        self.returns = o.form_data()
        self.portfolio_returns = None
        self.annual_factor = 12
        self.risk_free_rate = 0.02
        self.cumulative_returns = None
        self.portfolio_metrics = None
        self.quintiles=None
      

    def momentum_signal(self):
        momentum_signals = pd.DataFrame(index=self.returns.index, columns=self.returns.columns[1:-1])  # Exclude 'World, Date'
        momentum_signals['Date'] = self.returns['Date']  # Add Date column first
        for country in momentum_signals.columns[:-1]:  # Exclude 'Date'
            momentum_signals[country] = self.returns[country].rolling(window=12).apply(
                lambda x: x[:-1].sum() if len(x) == 12 else np.nan, raw=True)

        
        return momentum_signals
    
    def assign_to_portfolios(self, momentum):
        # Ranking increasingly in five equal sized groups
        # print(momentum)
        ranks = momentum.rank()
        self.quintiles = pd.qcut(ranks, 5, labels=False)
        return self.quintiles
    
    # Generates montly returns 

    def portfolio_generate(self):
        self.portfolio_returns = pd.DataFrame(index=self.returns.index, columns=['P1', 'P2', 'P3', 'P4', 'P5'])

        momentum_signals = self.momentum_signal()
        
        for date in momentum_signals.index[11:]:
            current_momentum = momentum_signals.loc[date].iloc[:-1]
            portfolios = self.assign_to_portfolios(current_momentum)
            next_month = date+1
            if next_month in self.returns.index:
                for q in range(0,5):
                    # print(portfolios[portfolios == q])
                    portfolio_countries = portfolios[portfolios == q].index
                    self.portfolio_returns.loc[next_month, f'P{q + 1}'] = self.returns.loc[next_month, portfolio_countries].mean()
                    # print(self.returns.loc[next_month, portfolio_countries].mean())
        

        self.portfolio_returns = self.portfolio_returns.dropna()
        return self.portfolio_returns

    # Part 3a
    # Annulising returns
    def set_annulised_mean_returns(self):

        self.portfolio_metrics = pd.DataFrame(index=self.portfolio_returns.columns, columns=['Mean Return', 'Standard Deviation', 'Sharpe Ratio'])

        for portfolio in self.portfolio_returns.columns:
            mean_return = self.portfolio_returns[portfolio].mean() * self.annual_factor
            std_dev = self.portfolio_returns[portfolio].std() * (self.annual_factor**0.5)
            sharpe_ratio = (mean_return - self.risk_free_rate) / std_dev
            self.portfolio_metrics.loc[portfolio] = [mean_return, std_dev, sharpe_ratio]

        self.cumulative_returns = (1 + self.portfolio_returns).cumprod()
        return self.portfolio_metrics
    
    def plot_cumulative_returns(self):
        if self.cumulative_returns is None:
            raise ValueError("Cumulative returns have not been calculated. Run set_annualized_metrics first.")

        plt.figure(figsize=(14, 8))

        for portfolio in self.cumulative_returns.columns:
            plt.plot(self.cumulative_returns.index, self.cumulative_returns[portfolio], label=portfolio)

        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.title('Cumulative Returns of Momentum-Sorted Portfolios')
        plt.legend()
        plt.grid(True)
        plt.show()
    

    # Part 3B

    def portfolio_w_HML(self):
        self.portfolio_returns['HML'] = self.portfolio_returns['P5'] - self.portfolio_returns['P1']
        self.returns['HML'] = (self.portfolio_returns['P5'] - self.portfolio_returns['P1'])
        self.returns['HML'] = self.returns['HML'].astype(np.float64)
        
        self.returns=self.returns.dropna()
        
        mean_return_hml = self.portfolio_returns['HML'].mean() * self.annual_factor
        std_dev_hml = self.portfolio_returns['HML'].std() * (self.annual_factor**0.5)
        sharpe_ratio_hml = (mean_return_hml - self.risk_free_rate) / std_dev_hml

        hml_metrics = pd.DataFrame({
            'Mean Return': [mean_return_hml],
            'Standard Deviation': [std_dev_hml],
            'Sharpe Ratio': [sharpe_ratio_hml]
        }, index=['HML'])

        self.portfolio_metrics = pd.concat([self.portfolio_metrics, hml_metrics])

        self.cumulative_returns['HML'] = (1 + self.portfolio_returns['HML']).cumprod()
        return self.portfolio_metrics

    def plot_portfolio_w_HML(self):
        if self.cumulative_returns is None or 'HML' not in self.cumulative_returns.columns:
            raise ValueError("Cumulative returns for HML have not been calculated. Run portfolio_w_HML first.")

        plt.figure(figsize=(14, 8))

        for portfolio in self.cumulative_returns.columns:
            plt.plot(self.cumulative_returns.index, self.cumulative_returns[portfolio], label=portfolio)
        
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.title('Cumulative Returns of Momentum-Sorted Portfolios Including HML')
        plt.legend()
        plt.grid(True)
        plt.show()

    def regression(self):
        model = 'HML ~ World'
        # print(self.returns.head(20))
        regression = ols(model, data=self.returns.drop(columns='Date')).fit()
        print(regression.summary())
        world_coefficient = regression.params['World']
        print(world_coefficient)

    def get_returns(self):
        return self.returns


        

    

# momentum = Momentum()
# momentum.momentum_signal()

# momentum.portfolio_generate()
# momentum.set_annulised_mean_returns()
# momentum.plot_cumulative_returns()

# print(momentum.portfolio_w_HML())
# momentum.plot_portfolio_w_HML()
# momentum.regression()
