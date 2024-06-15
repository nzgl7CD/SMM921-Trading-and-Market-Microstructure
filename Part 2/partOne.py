import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from IPython.display import display

class PartOne:
    def __init__(self) -> None:
        self.metric=None
        self.outlier=None
        self.file_path='Part 2\SMM921_pf_data_2024.xlsx'


    def get_returns(self):
        # Load the data from the provided Excel file
        data = pd.read_excel(self.file_path, sheet_name='Sheet1')

        # Convert the Date column to datetime format
        data['Date'] = pd.to_datetime(data['Date'])

        # Calculate monthly returns for each country
        returns = data.set_index('Date').pct_change().dropna()

        # Calculate the 'world stock market return' as the average return across the countries (excluding 'World')
        returns['World'] = returns.mean(axis=1)

        # Compute mean returns, return standard deviations and Sharpe ratios (all annualised) for each country
        annual_factor = 12
        mean_returns = returns.mean() * annual_factor
        std_devs = returns.std() * (annual_factor**0.5)
        risk_free_rate = 0.02  # Assume a risk-free rate of 2%
        sharpe_ratios = (mean_returns - risk_free_rate) / std_devs

        # Create a DataFrame to store the computed metrics
        self.metrics = pd.DataFrame({
            'Country': returns.columns,
            'Mean Return': mean_returns.values,
            'Standard Deviation': std_devs.values,
            'Sharpe Ratio': sharpe_ratios.values
        })

        # Set 'Country' as the index
        self.metrics.set_index('Country', inplace=True)
        
        print()
        display(self.metrics)
        print()

        # Exclude 'World' from the outlier calculations
        

        self.metrics.to_csv('Part 2/country_metrics.csv', index=True)
        returns.to_excel('Part 2\modified_SMM921_pf_data_2024.xlsx')
        return returns
    
    def outliers(self):
        metrics_without_world = self.metrics.drop('World')

        # Calculate means and standard deviations for outlier detection
        mean_return_avg = metrics_without_world['Mean Return'].mean()
        mean_return_std = metrics_without_world['Mean Return'].std()
        
        std_dev_avg = metrics_without_world['Standard Deviation'].mean()
        std_dev_std = metrics_without_world['Standard Deviation'].std()
        
        sharpe_ratio_avg = metrics_without_world['Sharpe Ratio'].mean()
        sharpe_ratio_std = metrics_without_world['Sharpe Ratio'].std()

        # Identify outliers
        outliers_mean = metrics_without_world[np.abs(metrics_without_world['Mean Return'] - mean_return_avg) > 2 * mean_return_std]
        outliers_std_dev = metrics_without_world[np.abs(metrics_without_world['Standard Deviation'] - std_dev_avg) > 2 * std_dev_std]
        outliers_sharpe = metrics_without_world[np.abs(metrics_without_world['Sharpe Ratio'] - sharpe_ratio_avg) > 2 * sharpe_ratio_std]

        # Combine all outliers into a single DataFrame
        self.outliers = pd.concat([outliers_mean, outliers_std_dev, outliers_sharpe]).drop_duplicates()

        # Display outliers
        print("Outliers:")
        display(self.outliers)
        print()
        return self.outliers
    
    def plot_data(self):
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot Mean Return and Standard Deviation on the primary y-axis
        ax1.set_xlabel('Company')
        ax1.set_ylabel('Mean Return and Standard Deviation')
        ax1.plot(self.metrics.index, self.metrics['Mean Return'], color='tab:blue', marker='o', label='Mean Return')
        ax1.plot(self.metrics.index, self.metrics['Standard Deviation'], color='tab:orange', marker='x', label='Standard Deviation')
        ax1.tick_params(axis='y')
        
        # Plot average lines for Mean Return and Standard Deviation
        mean_return_avg = self.metrics['Mean Return'].mean()
        std_dev_avg = self.metrics['Standard Deviation'].mean()
        ax1.axhline(mean_return_avg, color='tab:blue', linestyle='--', label='Mean Return Avg')
        ax1.axhline(std_dev_avg, color='tab:orange', linestyle='--', label='Standard Deviation Avg')
        
        ax1.legend(loc='upper left')

        # Create a secondary y-axis for Sharpe Ratio
        ax2 = ax1.twinx()
        ax2.set_ylabel('Sharpe Ratio')
        ax2.plot(self.metrics.index, self.metrics['Sharpe Ratio'], color='tab:green', marker='d', label='Sharpe Ratio')
        ax2.tick_params(axis='y')

        # Plot average line for Sharpe Ratio
        sharpe_ratio_avg = self.metrics['Sharpe Ratio'].mean()
        ax2.axhline(sharpe_ratio_avg, color='tab:green', linestyle='--', label='Sharpe Ratio Avg')

        ax2.legend(loc='upper right')

        # Improve the x-axis labels
        ax1.set_xticks(range(len(self.metrics.index)))
        ax1.set_xticklabels(self.metrics.index, rotation=90)

        # Title and layout adjustments
        plt.title('Company Metrics: Mean Return, Standard Deviation, and Sharpe Ratio')
        fig.tight_layout()  # To make sure labels and titles fit into the plot area
        plt.grid(True)
        plt.show()

  
# o=PartOne()
# o.get_returns()
# o.plot_data()
# o.outliers()
