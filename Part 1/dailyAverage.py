import pandas as pd
import matplotlib.pyplot as plt
from calculations import Calculations as Clc
import numpy as np
# import scipy.stats as sps
# import statsmodels.api as sm
import statsmodels.formula.api as smf
# import statsmodels.stats.api as sms
# import statsmodels.tsa.api as smt
# import statsmodels.stats.diagnostic as smdiag
# import statsmodels.stats.outliers_influence as oi
# import statsmodels.stats.stattools as stats
# import statsmodels.tsa.ar_model as sar

class DataProcessor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.prepare_data()

    # Task 3

    def prepare_data(self):
        # Choose one unique stock. i=0 is AAL in this dataset
        self.unique_stocks = self.dataset['Stock'].unique().tolist()
        self.dataset = self.dataset[self.dataset['Stock'] == self.unique_stocks[0]].dropna()
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date']).dt.date
        self.dataset['Time'] = pd.to_datetime(self.dataset['Time'], format='%H:%M:%S').dt.time

        # Calculate daily averages
        self.dataset['AbsMidQuoteReturn'] = self.dataset['MidQuoteReturn'].abs()
        
        self.dataset['DailyAvgMidQuoteVolatility'] = self.dataset.groupby(self.dataset['Date'])['AbsMidQuoteReturn'].transform('mean')
        self.dataset['DailyAvgSpread'] = self.dataset.groupby(self.dataset['Date'])['Spread'].transform('mean')
        self.dataset['DailyAvgDepth'] = self.dataset.groupby(self.dataset['Date'])['Depth'].transform('mean')
        

    # Plotting task 3a

    def plot_daily_series(self):
        specific_time = pd.to_datetime('16:25:00').time()
        filtered_dataset = self.dataset[self.dataset['Time'] == specific_time]

        # Check if there is data available
        if filtered_dataset.empty:
            print("No data available for the specified time '16:25:00'.")
            return

        # Plot the data for Daily Avg Spread
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(filtered_dataset['Date'], filtered_dataset['DailyAvgSpread'], label='Daily Avg Spread', marker='o', color='blue')
        plt.xlabel('Date')
        plt.ylabel('Spread')
        plt.title('Daily Average Spread at 16:25:00')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # Plot the data for Daily Avg Depth
        plt.subplot(3, 1, 2)
        plt.plot(filtered_dataset['Date'], filtered_dataset['DailyAvgDepth'], label='Daily Avg Depth', marker='o', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Depth')
        plt.title('Daily Average Depth at 16:25:00')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # Plot the data for Daily Avg MidQuote Volatility
        plt.subplot(3, 1, 3)
        plt.plot(filtered_dataset['Date'], filtered_dataset['DailyAvgMidQuoteVolatility'], label='Daily Avg MidQuote Volatility', marker='o', color='green')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.title('Daily Average MidQuote Volatility at 16:25:00')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()


    # Correlation avg depth and spread task 3b

    def get_correlation(self):
        return self.dataset['DailyAvgSpread'].corr(self.dataset['DailyAvgDepth'])

    def regression(self):
        formula = 'MidQuoteReturn ~ DailyAvgSpread'
        regressionSpread = smf.ols(formula, self.dataset).fit()
        model = 'MidQuoteReturn ~ DailyAvgDepth'
        regressionDepth=smf.ols(model, self.dataset).fit()
        return regressionSpread.summary(), '\n\n', regressionDepth.summary()

# Sample code


output_path = r'Part 1\modified_trading_data_2024.csv'
calc=Clc(output_path)
calc.set_datset()
calc.set_quote_spread()
calc.save_data()
ds=calc.get_dataset()


o=DataProcessor(ds)
# o.plot_daily_series()
print(o.get_correlation())
# o.plot_daily_series()
# print(f'Corr matrix \n {o.get_correlation()}')
