#!/usr/bin/env python
"""
Monte Carlo Stock Price Simulation with Spread Visualization
----------------------------------------------------------
This script simulates potential future stock price paths using Geometric Brownian Motion.
It generates multiple simulations, saves the data to CSV, and creates high-resolution plots
including all simulations, their mean, and visualizations of the spread over time.

Usage:
    The script can be run directly with default parameters or imported as a module.
    Default parameters simulate 1000 paths for a stock with initial price $50,
    daily drift of 0.0002, and daily volatility of 0.01 over 252 days (1 trading year).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Configure plotting style
plt.style.use("bmh")

# Constants
NUM_OF_SIMULATIONS = 10000  # Number of Monte Carlo simulations to run
DEFAULT_DAYS = 252         # Default simulation period (1 trading year)
PLOT_DPI = 300             # High resolution for saved figures
SPREAD_ALPHA = 0.2         # Transparency for spread visualization
SPREAD_PERCENTILES = [5, 25, 75, 95]  # Percentiles to calculate for spread

def create_directories():
    """Create required directories if they don't exist."""
    os.makedirs("./data/MCStocks", exist_ok=True)
    os.makedirs("./figs/MCStocks", exist_ok=True)

def stock_monte_carlo(S0, mu, sigma, N=DEFAULT_DAYS):
    """
    Run Monte Carlo simulation for stock price paths using Geometric Brownian Motion.
    
    Parameters:
    - S0: Initial stock price
    - mu: Daily expected return (drift)
    - sigma: Daily volatility
    - N: Number of days to simulate (default: 252)
    
    Returns:
    - simulation_data: DataFrame containing all simulations and summary statistics
    """
    result = []
    
    # Run multiple simulations
    for _ in range(NUM_OF_SIMULATIONS):
        prices = [S0]
        for _ in range(N):
            # Simulate price change using GBM formula
            stock_price = prices[-1] * np.exp(
                (mu - 0.5 * sigma ** 2) + sigma * np.random.normal()
            )
            prices.append(stock_price)
        result.append(prices)
    
    # Create DataFrame from results (time as rows, simulations as columns)
    simulation_data = pd.DataFrame(result).T
    
    # Calculate basic summary statistics
    simulation_data['mean'] = simulation_data.mean(axis=1)
    simulation_data['median'] = simulation_data.median(axis=1)
    
    # Calculate spread statistics
    for p in SPREAD_PERCENTILES:
        simulation_data[f'p{p}'] = simulation_data.iloc[:, :-2].apply(
            lambda x: np.percentile(x, p), axis=1
        )
    
    return simulation_data

def save_data(simulation_data):
    """Save simulation data to CSV file."""
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./data/MCStocks/mc_stock_sim_{timestamp}.csv"
    simulation_data.to_csv(filename, index_label="day")
    print(f"Simulation data saved to {filename}")

def plot_simulations(simulation_data, S0, mu, sigma):
    """
    Create and save plots of the Monte Carlo simulations with spread visualization.
    
    Parameters:
    - simulation_data: DataFrame containing simulation results
    - S0: Initial stock price (for title)
    - mu: Drift parameter (for title)
    - sigma: Volatility parameter (for title)
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Monte Carlo Stock Price Simulations\n(S0=${S0}, μ={mu:.4f}, σ={sigma:.4f}, {NUM_OF_SIMULATIONS} paths)')
    
    # Plot 1: All simulations with mean and percentiles
    for col in simulation_data.columns[:-6]:  # Exclude summary columns
        ax1.plot(simulation_data[col], color='blue', alpha=0.02, linewidth=0.5)
    
    # Plot important percentiles
    ax1.plot(simulation_data['mean'], color='red', linewidth=2, label='Mean')
    ax1.plot(simulation_data['median'], color='green', linewidth=2, linestyle='--', label='Median')
    
    # Fill between different percentile ranges
    ax1.fill_between(
        simulation_data.index,
        simulation_data['p5'],
        simulation_data['p95'],
        color='gray',
        alpha=SPREAD_ALPHA*0.7,
        label='90% Confidence'
    )
    ax1.fill_between(
        simulation_data.index,
        simulation_data['p25'],
        simulation_data['p75'],
        color='gray',
        alpha=SPREAD_ALPHA,
        label='50% Confidence'
    )
    
    ax1.set_ylabel('Stock Price ($)')
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.2f}'))
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # Plot 2: Spread visualization (range and IQR)
    ax2.plot(simulation_data['p95'] - simulation_data['p5'], 
             color='darkred', label='90% Range (p95-p5)')
    ax2.plot(simulation_data['p75'] - simulation_data['p25'], 
             color='darkblue', label='IQR (p75-p25)')
    
    ax2.set_xlabel('Trading Days')
    ax2.set_ylabel('Price Spread ($)')
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.2f}'))
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    
    # Save high-resolution plot
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"./figs/MCStocks/mc_stock_plot_{timestamp}.png"
    plt.savefig(plot_filename, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"Plot saved to {plot_filename}")
    
    plt.show()
    
    # Print final prediction with spread information
    final_mean = simulation_data['mean'].iloc[-1]
    final_median = simulation_data['median'].iloc[-1]
    final_p5 = simulation_data['p5'].iloc[-1]
    final_p95 = simulation_data['p95'].iloc[-1]
    
    print(f"\nPredicted future stock price after {DEFAULT_DAYS} days:")
    print(f"Mean: ${final_mean:,.2f}")
    print(f"Median: ${final_median:,.2f}")
    print(f"90% Confidence Interval: ${final_p5:,.2f} to ${final_p95:,.2f}")
    print(f"Price Range: ${simulation_data.iloc[-1, :-6].min():,.2f} to ${simulation_data.iloc[-1, :-6].max():,.2f}")

if __name__ == "__main__":
    # Create necessary directories
    create_directories()
    
    # Default simulation parameters
    INITIAL_PRICE = 50      # Starting stock price
    DAILY_DRIFT = 0.0002    # Daily expected return
    DAILY_VOLATILITY = 0.01 # Daily volatility
    
    # Run simulation
    sim_data = stock_monte_carlo(
        S0=INITIAL_PRICE,
        mu=DAILY_DRIFT,
        sigma=DAILY_VOLATILITY
    )
    
    # Save and plot results
    save_data(sim_data)
    plot_simulations(sim_data, INITIAL_PRICE, DAILY_DRIFT, DAILY_VOLATILITY)