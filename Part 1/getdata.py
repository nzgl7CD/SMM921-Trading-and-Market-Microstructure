import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class PartOne:
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.dataset = None
        self.unique_stocks=[]

    def get_data(self):
        self.dataset = pd.read_csv(self.file_path)
        return self.dataset

    def clean_up_data(self):
        if self.dataset is None:
            self.get_data()
            
            
        self.dataset['Time'] = self.dataset['Date-Time'].astype(str).str[11:19]
        self.dataset['Time']=pd.to_datetime(self.dataset['Time'], format='%H:%M:%S').dt.time
        

        cutoff_time_before_815 = pd.to_datetime('08:15:00', format='%H:%M:%S').time()
        cutoff_time_after_1625 = pd.to_datetime('16:25:00', format='%H:%M:%S').time()
        
        self.dataset = self.dataset[self.dataset['Time'] >= cutoff_time_before_815] # Filter the DataFrame to keep only rows where 'Time' is greater than or equal to cutoff_time
        self.dataset = self.dataset[self.dataset['Time'] <= cutoff_time_after_1625]

        self.dataset['Date'] = self.dataset['Date-Time'].astype(str).str[:10]
        self.dataset['Date and Time'] = self.dataset.apply(lambda row: f"{row['Date']} {row['Time']}", axis=1)

        return self.dataset
    
    def pick_three_stocks(self):
        self.unique_stocks = self.dataset['Stock'].unique().tolist()
        if len(self.unique_stocks) > 2:
            first_stock = self.unique_stocks[0]
            middle_stock = self.unique_stocks[len(self.unique_stocks) // 2]
            last_stock = self.unique_stocks[-1]
            selected_stocks = [first_stock, middle_stock, last_stock]
            
            # Filter the dataset to include only rows with the selected stocks
            self.dataset = self.dataset[self.dataset['Stock'].isin(selected_stocks)]
        else:
            # If there are less than 3 unique stocks, keep them all
            self.dataset = self.dataset[self.dataset['Stock'].isin(self.unique_stocks)]
        # print(self.dataset['Stock'].unique())

    def save_data(self, output_path):
        if self.dataset is not None:
            self.dataset.to_csv(output_path, index=False)
        else:
            print("No data to save. Please load and clean the data first.")

    def display_data(self):
        if self.dataset is not None:
            print(self.dataset)
        else:
            print("No data to display. Please load and clean the data first.")

# Usage example
# file_path = r'Part 1\SMM921_trading_data_2024.csv'
# output_path = r'Part 1\modified_trading_data_2024.csv'

# part_one = PartOne(file_path)
# part_one.get_data()
# part_one.clean_up_data()
# part_one.pick_three_stocks()
# part_one.save_data(output_path)
# part_one.display_data()

