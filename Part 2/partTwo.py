import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

class PortfolioAnalysis:
    def __init__(self):
        self.file_path = r'Part 2\SMM921_pf_data_2024.xlsx'
        self.dataset = None
        self.returns = None
        self.betas = {}

    def get_data(self):
        self.dataset = pd.read_excel(self.file_path)
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])
        
        return self.dataset

    def form_data(self):
        tempDates = pd.to_datetime(self.dataset['Date'])
        self.returns = self.dataset.drop(columns=['Date']).pct_change().dropna()
        # Calculate the 'world stock market return' as the average return across the countries
        self.returns['World'] = self.returns.mean(axis=1)
        
        # Combine the returns data with the date column
        self.returns.insert(0, 'Date', tempDates)
        
        return self.returns

    def save_data(self, output_path):
        if self.dataset is not None:
            self.dataset.to_excel(output_path, index=False)
        else:
            print("No data to save. Please load and clean the data first.")

    def calculate_betas(self):
        if self.returns is None:
            raise ValueError("Returns data not available. Run form_data() first.")

        world_return = self.returns['World']
        world_variance = world_return.var()

        # Calculate beta for each country
        for column in self.returns.columns:
            if column != 'World':
                covariance = self.returns[column].cov(world_return)
                beta = covariance / world_variance
                self.betas[column] = beta

    def plot_betas(self):
        betas_df = self.get_betas_dataframe()
        countries = betas_df['Country']
        betas = betas_df['Beta']

        plt.figure(figsize=(12, 8))
        plt.bar(countries, betas, color='skyblue')
        plt.xlabel('Country')
        plt.ylabel('Beta against World Return')
        plt.title('Country Betas against World Return')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.show()

    def interpret_betas(self):
        betas_df = self.get_betas_dataframe()

        print("Betas for each country:")
        print(betas_df)

        mean_beta = betas_df['Beta'].mean()
        std_beta = betas_df['Beta'].std()

        outliers = betas_df[np.abs(betas_df['Beta'] - mean_beta) > 2 * std_beta]
        
        if not outliers.empty:
            print("\nOutliers detected:")
            print(outliers)
        else:
            print("\nNo outliers detected.")

    def get_betas_dataframe(self):
        betas_df = pd.DataFrame(list(self.betas.items()), columns=['Country', 'Beta'])
        return betas_df

# Paths
output_path = r'Part 2\modified_SMM921_pf_data_2024.xlsx'


# Execution
# o = PortfolioAnalysis()
# o.get_data()
# o.form_data()
# o.form_data()
# o.calculate_betas()
# o.plot_betas()
# o.interpret_betas()

# The saver can be modified to save returns as well

# o.save_data(output_path)
