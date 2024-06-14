# from getdata import PartOne as PO
from calculations import Calculations as Clc
import pandas as pd
import matplotlib.pyplot as plt
# import statsmodels.stats.outliers_influence as oi
# import numpy as np
# import statsmodels.formula.api as smf
# import statsmodels.stats.api as sms

class Summeriser:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.dataset['Date-Time'] = pd.to_datetime(self.dataset['Date-Time'])
        self.dataset['Date'] = self.dataset['Date-Time'].dt.date
    # Task 1
    def summing(self):
        summary_spread = self.dataset.groupby('Stock')['Spread'].describe()
        summary_depth = self.dataset.groupby('Stock')['Depth'].describe()
        print("Spread Summary:\n", summary_spread)
        print("Depth Summary:\n", summary_depth)
    # Task 1
    def remove_outliers(self):
        # Identify and remove outliers using IQR method
        def iqr_outliers(data, column):
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        filtered_data = self.dataset.groupby('Stock').apply(lambda x: iqr_outliers(x, 'Spread')).reset_index(drop=True)
        filtered_data = filtered_data.groupby('Stock').apply(lambda x: iqr_outliers(x, 'Depth')).reset_index(drop=True)
        
        return filtered_data
    # Task 1
    def plot_time_series_spreads(self, remove_outliers=False):
        data_to_plot = self.remove_outliers() if remove_outliers else self.dataset
        daily_data = data_to_plot.groupby(['Date', 'Stock']).agg({'Spread': 'mean'}).reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))

        for stock in daily_data['Stock'].unique():
            stock_data = daily_data[daily_data['Stock'] == stock]
            ax.plot(stock_data['Date'], stock_data['Spread'], label=f'Spread - {stock}')

        ax.set_xlabel('Date')
        ax.set_ylabel('Spread', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_title('Daily Average Spread')
        ax.legend(loc='upper left')
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()
        plt.show()
    # Task 1
    def plot_time_series_depth(self, remove_outliers=False):
        data_to_plot = self.remove_outliers() if remove_outliers else self.dataset
        daily_data = data_to_plot.groupby(['Date', 'Stock']).agg({'Depth': 'mean'}).reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))

        for stock in daily_data['Stock'].unique():
            stock_data = daily_data[daily_data['Stock'] == stock]
            ax.plot(stock_data['Date'], stock_data['Depth'], label=f'Depth - {stock}')

        ax.set_xlabel('Date')
        ax.set_ylabel('Depth', color='orange')
        ax.tick_params(axis='y', labelcolor='orange')
        ax.set_title('Daily Average Depth')
        ax.legend(loc='upper left')
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()
        plt.show()
    # Task 2
    def mean_measures(self):
        # Convert 'Time' column to datetime if it's not already
        self.dataset['Time'] = pd.to_datetime(self.dataset['Time'], format='%H:%M:%S').dt.time

        # Filter dataset to include only rows where Time is hh:15:00
        hourly_data = self.dataset[(self.dataset['Time'].apply(lambda x: x.minute == 15)) & (self.dataset['Time'].apply(lambda x: x.second == 0))]

        # Group by Date, Stock, and hour (Time), calculate mean Spread and Depth
        hourly_means = hourly_data.groupby(['Date', 'Stock', self.dataset['Time'].apply(lambda x: x.hour)]).agg({
            'Spread': 'mean',
            'Depth': 'mean'
        }).reset_index()

        # Plotting daily hour means for each stock
        fig, axes = plt.subplots(nrows=len(hourly_means['Stock'].unique()), ncols=2, figsize=(16, 12), sharex=True)

        for i, stock in enumerate(hourly_means['Stock'].unique()):
            stock_data = hourly_means[hourly_means['Stock'] == stock]

            for j, metric in enumerate(['Spread', 'Depth']):
                ax = axes[i, j]
                for hour in stock_data['Time'].unique():
                    hour_data = stock_data[stock_data['Time'] == hour]
                    ax.plot(hour_data['Date'], hour_data[metric], label=f'{metric} at {hour}:15')
                
                ax.set_xlabel('Date')
                ax.set_ylabel(f'Mean {metric}')
                ax.set_title(f'Mean {metric} Variation for {stock}')
                ax.grid(True)
                ax.legend()

        plt.tight_layout()
        plt.show()

        return hourly_means

        # Plot mean Depth across hours for each stock
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

        for stock, ax in zip(hourly_means['Stock'].unique(), axes.flatten()):
            stock_data = hourly_means[hourly_means['Stock'] == stock]
            ax.plot(stock_data['Date'], stock_data['Depth'], label='Depth', marker='o', linestyle='--', color='orange')
            ax.set_xlabel('Date')
            ax.set_ylabel('Mean Depth')
            ax.set_title(f'Mean Depth Variation for {stock}')
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()

        return hourly_means

if __name__ == "__main__":
    output_path = r'Part 1\modified_trading_data_2024.csv'
    calc = Clc(output_path)
    calc.set_datset()
    calc.set_quote_spread()
    calc.save_data()
    ds = calc.get_dataset()

    summariser = Summeriser(ds)
    summariser.summing()
    # summariser.plot_time_series_spreads(remove_outliers=True)  
    # summariser.plot_time_series_depth(remove_outliers=True)   
    mean_measures = summariser.mean_measures()
    print(mean_measures)
