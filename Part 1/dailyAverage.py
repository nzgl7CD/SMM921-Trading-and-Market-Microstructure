from getdata import PartOne as PO
from calculations import Calculations as Clc
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.outliers_influence as oi
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

class DailyAvg:
    def __init__(self, dataset) -> None:
        
        self.dataset=dataset
        self.unique_stocks = None
    
    def set_data(self):
        self.unique_stocks = self.dataset['Stock'].unique().tolist()
        self.dataset = self.dataset[(self.dataset['Stock'] == self.unique_stocks[0]) & (self.dataset['Time'] == '16:25:00')]
        
    def daily_avg(self):
        average_values = self.dataset.groupby('Stock')[['Spread', 'Depth']].mean()

        average_values_df = average_values.reset_index()

        midquote_volatility = np.std(self.dataset['Mid Quote Return'])
        print(average_values_df, '\n\n', midquote_volatility)
    
    def plot_daily_series(self):
        self.dataset['Date-Time'] = pd.to_datetime(self.dataset['Date-Time'])

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot Spread on the primary y-axis
        ax1.plot(self.dataset['Date-Time'], self.dataset['Spread'], label='Spread', color='blue')
        ax1.set_xlabel('Date-Time')
        ax1.set_ylabel('Spread', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title('Time-Series of Spread and Depth')

        # Create a secondary y-axis to plot Depth
        ax2 = ax1.twinx()
        ax2.plot(self.dataset['Date-Time'], self.dataset['Depth'], label='Depth', color='orange')
        ax2.set_ylabel('Depth', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        # Add legends
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

        plt.tight_layout()
        plt.show()
    
    # Task 3b
    def get_correlation(self):
        return self.dataset['Spread'].corr(self.dataset['Depth'])
    
    def regression():
        return None
          
        


output_path = r'Part 1\modified_trading_data_2024.csv'
calc=Clc(output_path)
calc.set_datset()
calc.set_quote_spread()
calc.save_data()
ds=calc.get_dataset()

o=DailyAvg(ds)
o.set_data()
o.daily_avg()
# o.plot_daily_series()
print(f'Correlation {o.get_correlation()}')

        