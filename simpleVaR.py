#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Value at Risk (VaR) Calculator for Stocks

This script calculates the Value at Risk (VaR) for a given stock position using historical data.
It supports both single-day and multi-day VaR calculations under normal distribution assumptions.

Dependencies:
    numpy, pandas, yfinance, scipy
"""

import numpy as np
import yfinance as yf
from scipy.stats import norm
import datetime
import pandas as pd


def download_data(stock: str, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
    """
    Download stock price data from Yahoo Finance.
    
    Args:
        stock: Ticker symbol of the stock (e.g., 'C' for Citigroup)
        start_date: Start date for historical data
        end_date: End date for historical data
    
    Returns:
        DataFrame with the stock's adjusted close prices (or close prices if adjusted not available)
    """
    # Download data from Yahoo Finance
    ticker = yf.download(stock, start_date, end_date)
    
    # Handle MultiIndex case (some tickers return multi-level columns)
    if isinstance(ticker.columns, pd.MultiIndex):
        # Try to find 'Adj Close' in the MultiIndex
        if ('Adj Close', stock) in ticker.columns:
            return pd.DataFrame({stock: ticker[('Adj Close', stock)]})
        # Fall back to regular 'Close' if necessary
        else:
            print("Note: Adjusted Close not available, using Close price instead")
            return pd.DataFrame({stock: ticker[('Close', stock)]})
    else:
        # For single level columns
        try:
            return pd.DataFrame({stock: ticker['Adj Close']})
        except KeyError:
            print("Note: Adjusted Close not available, using Close price instead")
            return pd.DataFrame({stock: ticker['Close']})


def calculate_var(position: float, confidence: float, mu: float, sigma: float) -> float:
    """
    Calculate Value at Risk (VaR) for tomorrow (1-day VaR).
    
    Args:
        position: Dollar value of the investment
        confidence: Confidence level (e.g., 0.95 for 95%)
        mu: Mean of daily returns
        sigma: Standard deviation of daily returns
    
    Returns:
        VaR in dollars
    """
    var = position * (mu - sigma * norm.ppf(1 - confidence))
    return var


def calculate_var_n(position: float, confidence: float, mu: float, sigma: float, days: int) -> float:
    """
    Calculate Value at Risk (VaR) for n days in the future.
    
    Args:
        position: Dollar value of the investment
        confidence: Confidence level (e.g., 0.95 for 95%)
        mu: Mean of daily returns
        sigma: Standard deviation of daily returns
        days: Number of days to calculate VaR for
    
    Returns:
        VaR in dollars for the given time horizon
    """
    var = position * (mu * days - sigma * np.sqrt(days) * norm.ppf(1 - confidence))
    return var


if __name__ == "__main__":
    # Set date range for historical data
    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2018, 1, 1)
    
    # Download stock data (using 'C' for Citigroup)
    stock_data = download_data('C', start, end)
    
    # Calculate daily log returns
    stock_data['returns'] = np.log(stock_data['C'] / stock_data['C'].shift(1))
    stock_data = stock_data[1:]  # Drop first row with NaN
    
    print("First 5 rows of stock data with returns:")
    print(stock_data.head())
    
    # Investment parameters
    investment = 1e6  # $1,000,000 investment
    confidence_level = 0.95  # 95% confidence level
    
    # Calculate return statistics
    mu = np.mean(stock_data['returns'])
    sigma = np.std(stock_data['returns'])
    
    print("\nRisk Analysis Results:")
    # Calculate and print 1-day VaR
    var_tomorrow = calculate_var(investment, confidence_level, mu, sigma)
    print(f"VaR Tomorrow (maximum loss at 95% confidence): ${var_tomorrow:,.2f}")
    
    # Calculate and print 10-day VaR
    var_10_days = calculate_var_n(investment, confidence_level, mu, sigma, 10)
    print(f"VaR 10 days (maximum loss at 95% confidence): ${var_10_days:,.2f}")