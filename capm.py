import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os
plt.style.use("bmh")

# market interest rate
RISK_FREE_RATE = 0.05
# consider annual return 
MONTHS_IN_YEAR = 12

# Create directories if they don't exist
os.makedirs('./data/CAPM', exist_ok=True)
os.makedirs('./figs/CAPM', exist_ok=True)

class CAPM:
    def __init__(self, stocks, start_date, end_date):
        self.data = None
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        
    def download_data(self):
        # Create an empty DataFrame first
        data = pd.DataFrame()
        
        for stock in self.stocks:
            # Set auto_adjust=False to get the 'Adj Close' column
            ticker = yf.download(stock, self.start_date, self.end_date, auto_adjust=False)
            # Add the stock's adjusted close prices as a column to the DataFrame
            data[stock] = ticker['Adj Close']
        
        # Save the raw downloaded data
        data.to_csv('./data/CAPM/raw_data.csv')
        print(f"Raw data saved to ./data/CAPM/raw_data.csv")
        
        return data
        
    def initialize(self):
        self.data = self.download_data()
        
        # we use monthly returns instead of daily returns
        self.data = self.data.resample('M').last()
        
        # print confirmation
        print("Data initialized successfully")
        print(self.data.head())
        
        # Rename the columns in self.data instead of creating a new DataFrame
        self.data = self.data.rename(columns={
            self.stocks[0]: 's_adjclose',
            self.stocks[1]: 'm_adjclose'
        })
        
        # log monthly returns
        self.data[['s_returns', 'm_returns']] = np.log(self.data[['s_adjclose', 'm_adjclose']] / self.data[['s_adjclose', 'm_adjclose']].shift(1))
        self.data = self.data[1:]
        
        # Save the processed data
        self.data.to_csv('./data/CAPM/processed_data.csv')
        print(f"Processed data saved to ./data/CAPM/processed_data.csv")
        
    def calculate_beta(self):
        # covariance matrix: the diag. items are the variances
        # off diagonals are covariances
        # the matrix is symmetric: cov[0,1] = cov[1,0]
        covariance_matrix = np.cov(self.data["s_returns"], self.data["m_returns"])
        # calculating beta according to the formula
        beta = covariance_matrix[0,1] / covariance_matrix[1,1]
        print("Beta from formula: ", beta)
        
        # Save the beta calculation results as CSV
        beta_df = pd.DataFrame({
            'method': ['covariance'],
            'beta': [beta],
            'cov_stock_market': [covariance_matrix[0,1]],
            'var_market': [covariance_matrix[1,1]]
        })
        beta_df.to_csv('./data/CAPM/beta_calculation.csv', index=False)
        print(f"Beta calculation saved to ./data/CAPM/beta_calculation.csv")
    
    def regresion(self):
        # using linear regression to fit a line to the data
        # [stock_returns, market_returns] -> slope is the beta
        beta, alpha = np.polyfit(self.data['m_returns'], self.data['s_returns'], deg=1)
        print("Beta from regression: ", beta)
        expected_return = RISK_FREE_RATE + beta * (self.data['m_returns'].mean()*MONTHS_IN_YEAR - RISK_FREE_RATE) 
        print("Expected return: ", expected_return)
        
        # Save the regression results as CSV
        results_df = pd.DataFrame({
            'metric': ['beta', 'alpha', 'expected_return'],
            'value': [beta, alpha, expected_return]
        })
        results_df.to_csv('./data/CAPM/regression_results.csv', index=False)
        print(f"Regression results saved to ./data/CAPM/regression_results.csv")
        
        # Also save fitted values for plotting
        fitted_df = pd.DataFrame({
            'm_returns': self.data['m_returns'],
            's_returns': self.data['s_returns'],
            'fitted_values': beta * self.data['m_returns'] + alpha
        })
        fitted_df.to_csv('./data/CAPM/fitted_values.csv', index=False)
        print(f"Fitted values saved to ./data/CAPM/fitted_values.csv")
        
        self.plot_regression(alpha, beta)
        
    def plot_regression(self, alpha, beta):
        fig, axis = plt.subplots(1, figsize=(20, 10))
        axis.scatter(self.data["m_returns"], self.data['s_returns'],
                     label="Data Points")
        axis.plot(self.data["m_returns"], beta * self.data["m_returns"] + alpha,
                  color='red', label="CAPM Line")
        plt.title('Capital Asset Pricing Model, finding alpha and beta')
        plt.xlabel('Market return $R_m$', fontsize=18)
        plt.ylabel('Stock return $R_a$')
        plt.text(0.08, 0.05, r'$R_a = \beta * R_m + \alpha$', fontsize=18)
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig('./figs/CAPM/regression_plot.png', dpi=300, bbox_inches='tight')
        print(f"Regression plot saved to ./figs/CAPM/regression_plot.png")
        
        # Display the plot as well
        plt.show()
        
if __name__ == '__main__':
    capm = CAPM(['IBM', '^GSPC'], '2010-01-01', '2017-01-01')
    capm.initialize()
    print("\n")
    capm.calculate_beta()
    capm.regresion()