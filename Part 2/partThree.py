from partTwo import PortfolioAnalysis
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from partOne import PartOne
from statsmodels.formula.api import ols
from exogenous import Exogenous
from IPython.display import display

class Momentum:
    def __init__(self) -> None:
        o = PortfolioAnalysis()
        self.exo=Exogenous()
        self.dataset = o.get_data()
        self.returns = o.form_data()
        self.portfolio_returns = None
        self.annual_factor = self.exo.get_annual_factor()
        self.risk_free_rate = self.exo.get_risk_free_rate()
        self.cumulative_returns = None
        self.portfolio_metrics = None
        self.quintiles=None
        self.get_hml_metric=None
        self.set_annulised_mean_returns
        
    def momentum_signal(self):
        momentum_signals = pd.DataFrame(index=self.returns.index, columns=self.returns.columns[1:-1])  # Exclude 'World, Date'
        momentum_signals['Date'] = self.returns['Date']  # Add Date column first 
        for country in momentum_signals.columns[:-1]:  # Exclude 'Date'
            momentum_signals[country] = self.returns[country].rolling(window=12).apply(
                lambda x: x[:-1].sum() if len(x) == 12 else np.nan, raw=True)
        # display(momentum_signals[11:])
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

        # for portfolio in self.portfolio_returns.columns:
            # use (1+mean_return)**12-1
        mean_return = (1+self.portfolio_returns).prod()**(self.annual_factor/len(self.portfolio_returns))-1
        std_dev = self.portfolio_returns.std() * (self.annual_factor**0.5)
        sharpe_ratio = (mean_return - self.risk_free_rate) / std_dev
        self.portfolio_metrics.loc['P1'] = [mean_return['P1'], std_dev['P1'], sharpe_ratio['P1']]
        self.portfolio_metrics.loc['P2'] = [mean_return['P2'], std_dev['P2'], sharpe_ratio['P2']]
        self.portfolio_metrics.loc['P3'] = [mean_return['P3'], std_dev['P3'], sharpe_ratio['P3']]
        self.portfolio_metrics.loc['P4'] = [mean_return['P4'], std_dev['P4'], sharpe_ratio['P4']]
        self.portfolio_metrics.loc['P5'] = [mean_return['P5'], std_dev['P5'], sharpe_ratio['P5']]

        self.cumulative_returns = (1 + self.portfolio_returns).cumprod()
        print(f'portfolio_metrics: {self.portfolio_metrics}')
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
        print((1+self.portfolio_returns['HML']).prod())
        mean_return_hml = (1+self.portfolio_returns['HML']).prod() **(self.annual_factor/len(self.portfolio_returns['HML']))-1
        std_dev_hml = self.portfolio_returns['HML'].std() * (self.annual_factor**0.5)
        sharpe_ratio_hml = (mean_return_hml - self.risk_free_rate) / std_dev_hml

        self.hml_metrics = pd.DataFrame({
            'Mean Return': [mean_return_hml],
            'Standard Deviation': [std_dev_hml],
            'Sharpe Ratio': [sharpe_ratio_hml]
        }, index=['HML'])

        print(pd.concat([self.hml_metrics]))

        self.portfolio_metrics = pd.concat([self.portfolio_metrics, self.hml_metrics])

        self.cumulative_returns['HML'] = (1 + self.portfolio_returns['HML']).cumprod()
        print(self.portfolio_metrics)
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
    def get_portfolio_metric(self):
        return self.portfolio_metrics
        

    
if __name__ == "__main__":

    momentum = Momentum()
    # print(momentum.momentum_signal()[11:])
    momentum.momentum_signal()
    # momentum.portfolio_generate()
    # momentum.set_annulised_mean_returns()
    # momentum.portfolio_w_HML()
    # momentum.plot_portfolio_w_HML()




