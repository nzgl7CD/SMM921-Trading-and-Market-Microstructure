import pandas as pd
from getdata import PartOne as p1
class Calculations:
    def __init__(self, file_path) -> None:
        
        self.file_path=file_path
        self.dataset=None
    def set_datset(self):
        self.dataset = pd.read_csv(self.file_path)
    def get_dataset(self):
        return self.dataset
    
    def set_quote_spread(self):
        self.dataset['Mid Quote']=0.5*(self.dataset['Close Bid']+self.dataset['Close Ask']) ###Mid Quote as 1/2*(pb+pa)
        self.dataset['MidQuoteReturn'] = self.dataset['Mid Quote'].pct_change() ###Mid Quote Return as (mq1-mq0)/mq0 from index 1 to n as we don't calculate return for the first value
        self.dataset['Spread'] = 10000*((self.dataset['Close Ask']-self.dataset['Close Bid'])/self.dataset['Mid Quote']) ###Spread as 10k*(pa-pb)/mq
        # self.dataset['Depth'] = 0.5*(self.dataset['Close Bid']*self.dataset['Close Bid Size']+self.dataset['Close Ask']*self.dataset['Close Ask Size']) ###depth as 1/2*(Qa*Pa+Qb*Pb)
        self.dataset['Depth'] = 0.5*(self.dataset['Close Bid Size']+self.dataset['Close Ask Size']) ###depth as 1/2*(Qa*Pa+Qb*Pb)

        self.dataset = self.dataset[self.dataset['Spread'] >= 0]
    
    def save_data(self):
        if self.dataset is not None:
            self.dataset.to_csv(self.file_path, index=False)
        else:
            print("No data to save. Please load and clean the data first.")

    def display_data(self):
        if self.dataset is not None:
            print(self.dataset)
        else:
            print("No data to display. Please load and clean the data first.")

    



# output_path = r'Part 1\modified_trading_data_2024.csv'

# o = Calculations(output_path)
# o.set_datset()
# o.set_quote_spread()
# o.save_data()
# o.display_data()
