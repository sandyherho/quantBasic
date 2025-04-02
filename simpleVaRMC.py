#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo Value at Risk (VaR) Calculator for Stocks
This script calculates the Value at Risk (VaR) using Monte Carlo simulation.
"""
import numpy as np
import yfinance as yf
from scipy.stats import norm
import datetime
import pandas as pd

def download_data(stock, start, end):
    """
    Download stock price data from Yahoo Finance.
    
    Args:
        stock: Ticker symbol (e.g., 'C' for Citigroup)
        start: Start date
        end: End date
    
    Returns:
        DataFrame with closing prices
    """
    ticker = yf.download(stock, start, end)
    
    # Check if data was successfully downloaded
    if ticker.empty:
        raise ValueError(f"No data found for {stock} between {start} and {end}")
        
    # Return the Close price column directly
    return ticker[['Close']]  # Return as a DataFrame with only the Close column

class ValueAtRiskMonteCarlo:
    def __init__(self, S, mu, sigma, c, n, iterations):
        """
        Initialize Monte Carlo VaR calculator.
        
        Args:
            S: Investment value at t=0 (e.g., $1,000,000)
            mu: Mean of daily returns
            sigma: Standard deviation of daily returns
            c: Confidence level (e.g., 0.95 for 95%)
            n: Time horizon in days
            iterations: Number of Monte Carlo simulations
        """
        self.S = S
        self.mu = mu
        self.sigma = sigma
        self.c = c
        self.n = n
        self.iterations = iterations
        
    def simulation(self):
        """
        Run Monte Carlo simulation to calculate VaR.
        
        Returns:
            Value at Risk in dollars
        """
        # Generate random numbers from normal distribution
        rand = np.random.normal(0, 1, [1, self.iterations])
        
        # Calculate stock price paths using geometric Brownian motion
        stock_price = self.S * np.exp(self.n * (self.mu - 0.5 * self.sigma ** 2) +
                                     self.sigma * np.sqrt(self.n) * rand)
        
        # Sort the simulated prices to determine percentile
        stock_price = np.sort(stock_price)
        
        # Calculate the percentile based on confidence level
        percentile = np.percentile(stock_price, (1 - self.c) * 100)
        
        return self.S - percentile

if __name__ == "__main__":
    # Parameters
    S = 1e6  # Investment amount ($1,000,000)
    c = 0.95  # 95% confidence level
    n = 1  # 1 day VaR
    iterations = 100000  # Number of Monte Carlo simulations
    
    # Historical data range
    start_date = datetime.datetime(2014, 1, 1)
    end_date = datetime.datetime(2017, 10, 15)
    
    # Download stock data
    citi = download_data('C', start_date, end_date)
    print("Data downloaded successfully:")
    print(citi.head())
    
    # Calculate daily returns
    citi['returns'] = citi['Close'].pct_change()
    citi = citi.dropna()  # Remove NA values from returns
    
    # Calculate statistics
    mu = np.mean(citi['returns'])
    sigma = np.std(citi['returns'])
    
    print(f"\nCalculated statistics:")
    print(f"Mean daily return: {mu:.6f}")
    print(f"Daily volatility: {sigma:.6f}")
    
    # Run Monte Carlo simulation
    model = ValueAtRiskMonteCarlo(S, mu, sigma, c, n, iterations)
    var = model.simulation()
    
    print(f"\nValue at Risk Results:")
    print(f"1-day VaR at {c*100:.0f}% confidence level: ${var:,.2f}")