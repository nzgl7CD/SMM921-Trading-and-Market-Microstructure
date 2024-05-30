from getdata import PartOne as PO
from calculations import Calculations as Clc
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.outliers_influence as oi
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

class Summeriser:

    def __init__(self, dataset) -> None:
        
        self.dataset=dataset
    

    def summing(self):
        # summary = self.dataset.groupby('Stock')[['Spread', 'Depth']].describe()
        summary = self.dataset[['Spread', 'Depth']].describe()
        return summary
    # Should we remove outlier?
    
    def plot_time_series(self):

    # Ensure 'Date-Time' is in datetime format
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
    
    def mean_measures(self):
        
        dictionary={'Stock':[],'Spread':[],'Depth':[]}
        
        self.dataset['Time'] = pd.to_datetime(self.dataset['Time'], format='%H:%M:%S').dt.time
        #    average_values = self.dataset.groupby('Stock')[['Spread', 'Depth']].mean()
        # print(average_values)

        # Loop through every 60th row
        for index, row in self.dataset.iterrows():
            # Check if the 'Time' value is on the hour
            if row['Time'].minute == 0 and row['Time'].second == 0:
                dictionary['Stock'].append(self.dataset['Stock'].iloc[index])
                dictionary['Spread'].append(self.dataset['Spread'].iloc[index])
                dictionary['Depth'].append(self.dataset['Depth'].iloc[index])
        df=pd.DataFrame(dictionary)
        mean_values = df.groupby('Stock').mean()
        return mean_values
        
        


output_path = r'Part 1\modified_trading_data_2024.csv'
calc=Clc(output_path)
calc.set_datset()
calc.set_quote_spread()
calc.save_data()
ds=calc.get_dataset()

o=Summeriser(ds)
sumer=o.summing()
means=o.mean_measures()
print(sumer, '\n', means)


