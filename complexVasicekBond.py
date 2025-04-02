#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vasicek Short Rate Model for Bond Pricing via Monte Carlo Simulation

This script implements the Vasicek model to simulate short-term interest rates and
price zero-coupon bonds using Monte Carlo methods. The Vasicek model is an 
Ornstein-Uhlenbeck process that allows mean reversion in interest rates.

The stochastic differential equation is:
    dr(t) = κ(θ - r(t))dt + σdW(t)

where:
    κ = speed of mean reversion
    θ = long-term mean level
    σ = volatility of the short rate
    W(t) = Wiener process (standard Brownian motion)

Author: Sandy Herho <sandy.herho@email.ucr.edu>
Date: March 28, 2025
License: MIT
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from tqdm import tqdm

# Set plotting style
plt.style.use("bmh")
sns.set_context("paper")

# Simulation parameters
NUM_OF_SIMULATIONS = 1000  # Number of Monte Carlo simulations
NUM_OF_POINTS = 10000      # Number of time steps in each simulation

def ensure_directories_exist():
    """
    Create output directories for figures and data if they don't exist.
    """
    directories = ["./figs/vasicekBond", "./data/vasicekBond"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def monte_carlo_simulation(face_value, r0, kappa, theta, sigma, T=1, plot=True):
    """
    Perform Monte Carlo simulation of the Vasicek model and calculate bond price.
    
    Parameters:
    -----------
    face_value : float
        The face value of the zero-coupon bond.
    r0 : float
        Initial short rate.
    kappa : float
        Mean reversion speed.
    theta : float
        Long-term mean level.
    sigma : float
        Volatility of the short rate.
    T : float, optional
        Time to maturity in years. Default is 1.
    plot : bool, optional
        Whether to generate and save plots. Default is True.
        
    Returns:
    --------
    bond_price : float
        The calculated bond price.
    simulation_data : pandas.DataFrame
        Dataframe containing all simulated paths.
    """
    dt = T / float(NUM_OF_POINTS)
    time_points = np.linspace(0, T, NUM_OF_POINTS + 1)
    result = []
    
    # Progress bar for simulation
    print(f"Running {NUM_OF_SIMULATIONS} Monte Carlo simulations with {NUM_OF_POINTS} time steps each...")
    for i in tqdm(range(NUM_OF_SIMULATIONS)):
        rates = [r0]
        for j in range(NUM_OF_POINTS):
            # Vasicek SDE discretization (Euler-Maruyama method)
            dr = kappa * (theta - rates[-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
            rates.append(rates[-1] + dr)
        result.append(rates)
    
    # For efficiency when plotting, downsample to 1000 points if NUM_OF_POINTS is large
    if NUM_OF_POINTS > 1000:
        downsample_factor = NUM_OF_POINTS // 1000
        time_points_plot = time_points[::downsample_factor]
        result_plot = [path[::downsample_factor] for path in result]
        plot_data = pd.DataFrame(result_plot).T
        plot_data.index = time_points_plot
    else:
        plot_data = pd.DataFrame(result).T
        plot_data.index = time_points
        
    # Full resolution data for calculations
    simulation_data = pd.DataFrame(result).T
    simulation_data.index = time_points
    
    # Save the simulation data
    simulation_data.to_csv("./data/vasicekBond/simulation_data.csv")
    
    # Calculate statistics for each time point
    mean_rates = simulation_data.mean(axis=1)
    median_rates = simulation_data.median(axis=1)
    std_rates = simulation_data.std(axis=1)
    
    # Save statistics
    stats_df = pd.DataFrame({
        'time': time_points,
        'mean_rate': mean_rates,
        'median_rate': median_rates,
        'std_dev': std_rates,
        'min_rate': simulation_data.min(axis=1),
        'max_rate': simulation_data.max(axis=1),
        '5th_percentile': simulation_data.quantile(0.05, axis=1),
        '95th_percentile': simulation_data.quantile(0.95, axis=1)
    })
    stats_df.to_csv("./data/vasicekBond/rate_statistics.csv", index=False)
    
    # Calculate the integral of r(t) for each path using trapezoidal rule
    # This is more accurate than simple sum
    integral_sum = np.zeros(NUM_OF_SIMULATIONS)
    for i in range(NUM_OF_SIMULATIONS):
        integral_sum[i] = np.trapz(simulation_data.iloc[:, i], time_points)
    
    # Present value of future cash flow
    present_integral_sum = np.exp(-integral_sum)
    
    # Calculate bond price (mean of present values)
    bond_price = face_value * np.mean(present_integral_sum)
    
    # Calculate bond prices at different confidence levels
    confidence_levels = [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]
    bond_prices_at_confidence = {
        f"{int(cl*100)}%": face_value * np.percentile(present_integral_sum, cl*100)
        for cl in confidence_levels
    }
    
    # Save bond price statistics
    price_stats = {
        'bond_price': bond_price,
        'bond_price_std': face_value * np.std(present_integral_sum),
        'bond_price_min': face_value * np.min(present_integral_sum),
        'bond_price_max': face_value * np.max(present_integral_sum),
        **bond_prices_at_confidence
    }
    pd.DataFrame([price_stats]).to_csv("./data/vasicekBond/bond_price_statistics.csv", index=False)
    
    # Print results
    print("\nVasicek Model Parameters:")
    print(f"Initial Rate (r0): {r0:.4f}")
    print(f"Mean Reversion Speed (κ): {kappa:.4f}")
    print(f"Long-term Mean (θ): {theta:.4f}")
    print(f"Volatility (σ): {sigma:.4f}")
    print(f"Time to Maturity: {T} year(s)")
    print(f"Face Value: ${face_value:.2f}")
    print("\nBond Price Results:")
    print(f"Bond price based on Monte-Carlo simulation: ${bond_price:.2f}")
    print(f"Standard Deviation: ${face_value * np.std(present_integral_sum):.2f}")
    print(f"95% Confidence Interval: ${bond_prices_at_confidence['5%']:.2f} - ${bond_prices_at_confidence['95%']:.2f}")
    
    if plot:
        # Create and save plots
        plot_simulation_paths(simulation_data, mean_rates, theta, r0, kappa, sigma, T)
        plot_rate_distribution(simulation_data, T)
        plot_bond_price_distribution(present_integral_sum, face_value, bond_price)
    
    return bond_price, simulation_data

def plot_simulation_paths(simulation_data, mean_rates, theta, r0, kappa, sigma, T):
    """
    Plot the simulated interest rate paths.
    
    Parameters:
    -----------
    simulation_data : pandas.DataFrame
        Dataframe containing all simulated paths.
    mean_rates : pandas.Series
        Mean interest rate at each time point.
    theta : float
        Long-term mean level.
    r0 : float
        Initial short rate.
    kappa : float
        Mean reversion speed.
    sigma : float
        Volatility of the short rate.
    T : float
        Time to maturity in years.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot a sample of paths (fewer paths for clarity - only 30)
    sample_cols = np.random.choice(simulation_data.columns, size=min(50, NUM_OF_SIMULATIONS), replace=False)
    # Use plot directly instead of DataFrame.plot to avoid column labels appearing
    for col in sample_cols:
        ax.plot(simulation_data.index, simulation_data[col], color='gray', alpha=0.08, linewidth=0.7)
    
    # Plot mean and confidence interval with fewer lines
    ax.plot(simulation_data.index, mean_rates, 'b-', linewidth=2.5, label='Mean Rate')
    
    # Create a confidence interval fill area instead of separate lines
    ci_lower = simulation_data.quantile(0.05, axis=1)
    ci_upper = simulation_data.quantile(0.95, axis=1)
    ax.fill_between(simulation_data.index, ci_lower, ci_upper, color='blue', alpha=0.1, label='90% Confidence Interval')
    
    # Add a horizontal line for the long-term mean
    ax.axhline(y=theta, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label=f'Long-term Mean (θ)')
    
    # Add arrow pointing to the long-term mean with annotation
    ax.annotate(f'θ = {theta:.2%}', xy=(T*0.9, theta), xytext=(T*0.9, theta+0.01),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='red'))
    
    # Format axes
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # Maximum of 10 ticks on x-axis
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Set labels and title with enhanced formatting
    ax.set_xlabel('Time (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Short Rate r(t)', fontsize=12, fontweight='bold')
    ax.set_title(f'Vasicek Model: Short Rate Evolution Over {T} Years\nParameters: r₀={r0:.2%}, κ={kappa:.2f}, θ={theta:.2%}, σ={sigma:.2%}', 
                fontsize=14, fontweight='bold')
    
    # More compact legend
    legend = ax.legend(loc='upper right', frameon=True, fontsize=10)
    legend.get_frame().set_alpha(0.7)
    
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Add text box with statistics (more compact)
    textstr = (f'Number of simulations: {NUM_OF_SIMULATIONS:,}\nInitial rate: {r0:.2%} → Final mean: {mean_rates.iloc[-1]:.2%} (σ={simulation_data.iloc[-1, :].std():.2%})')
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7, edgecolor='gray')
    ax.text(0.03, 0.03, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='left', bbox=props)
    
    plt.tight_layout()
    plt.savefig('./figs/vasicekBond/interest_rate_paths.png', dpi=300)
    plt.savefig('./figs/vasicekBond/interest_rate_paths.pdf')
    plt.close()

def plot_rate_distribution(simulation_data, T):
    """
    Plot the distribution of interest rates at maturity.
    
    Parameters:
    -----------
    simulation_data : pandas.DataFrame
        Dataframe containing all simulated paths.
    T : float
        Time to maturity in years.
    """
    final_rates = simulation_data.iloc[-1, :]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram with density curve
    sns.histplot(final_rates, kde=True, stat="density", bins=30, color='skyblue', ax=ax)
    
    # Add vertical lines for statistics
    plt.axvline(final_rates.mean(), color='red', linestyle='-', label=f'Mean: {final_rates.mean():.3f}')
    plt.axvline(final_rates.median(), color='green', linestyle='--', label=f'Median: {final_rates.median():.3f}')
    plt.axvline(np.percentile(final_rates, 5), color='purple', linestyle=':', label=f'5th Percentile: {np.percentile(final_rates, 5):.3f}')
    plt.axvline(np.percentile(final_rates, 95), color='purple', linestyle=':', label=f'95th Percentile: {np.percentile(final_rates, 95):.3f}')
    
    # Set labels and title
    ax.set_xlabel('Interest Rate at Maturity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Distribution of Interest Rates at T={T}', fontsize=14)
    
    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Add legend and grid
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add text box with statistics
    textstr = '\n'.join((
        f'Mean: {final_rates.mean():.2%}',
        f'Median: {final_rates.median():.2%}',
        f'Std Dev: {final_rates.std():.2%}',
        f'Skewness: {final_rates.skew():.3f}',
        f'Kurtosis: {final_rates.kurtosis():.3f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('./figs/vasicekBond/rate_distribution.png', dpi=300)
    plt.savefig('./figs/vasicekBond/rate_distribution.pdf')
    plt.close()

def plot_bond_price_distribution(present_values, face_value, bond_price):
    """
    Plot the distribution of bond prices.
    
    Parameters:
    -----------
    present_values : numpy.ndarray
        Array of discount factors from each simulation.
    face_value : float
        The face value of the bond.
    bond_price : float
        The calculated bond price.
    """
    bond_prices = face_value * present_values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram with density curve
    sns.histplot(bond_prices, kde=True, stat="density", bins=30, color='lightgreen', ax=ax)
    
    # Add vertical lines for statistics
    plt.axvline(bond_price, color='red', linestyle='-', label=f'Mean: ${bond_price:.2f}')
    plt.axvline(np.median(bond_prices), color='blue', linestyle='--', 
                label=f'Median: ${np.median(bond_prices):.2f}')
    plt.axvline(np.percentile(bond_prices, 5), color='purple', linestyle=':', 
                label=f'5th Percentile: ${np.percentile(bond_prices, 5):.2f}')
    plt.axvline(np.percentile(bond_prices, 95), color='purple', linestyle=':', 
                label=f'95th Percentile: ${np.percentile(bond_prices, 95):.2f}')
    
    # Set labels and title
    ax.set_xlabel('Bond Price ($)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Zero-Coupon Bond Prices', fontsize=14)
    
    # Add legend and grid
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add text box with statistics
    textstr = '\n'.join((
        f'Mean: ${bond_prices.mean():.2f}',
        f'Median: ${np.median(bond_prices):.2f}',
        f'Std Dev: ${bond_prices.std():.2f}',
        f'95% CI: (${np.percentile(bond_prices, 5):.2f}, ${np.percentile(bond_prices, 95):.2f})',
        f'Discount: {100 * (1 - bond_price / face_value):.2f}%'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('./figs/vasicekBond/bond_price_distribution.png', dpi=300)
    plt.savefig('./figs/vasicekBond/bond_price_distribution.pdf')
    plt.close()

def calculate_analytical_price(face_value, r0, kappa, theta, sigma, T):
    """
    Calculate the analytical bond price using the closed-form solution of the Vasicek model.
    
    Parameters:
    -----------
    face_value : float
        The face value of the zero-coupon bond.
    r0 : float
        Initial short rate.
    kappa : float
        Mean reversion speed.
    theta : float
        Long-term mean level.
    sigma : float
        Volatility of the short rate.
    T : float
        Time to maturity in years.
        
    Returns:
    --------
    float
        The analytical bond price.
    """
    B = (1 - np.exp(-kappa * T)) / kappa
    A = np.exp((theta - (sigma**2) / (2 * kappa**2)) * (B - T) - (sigma**2) / (4 * kappa) * B**2)
    price = A * np.exp(-B * r0) * face_value
    return price

def compare_methods(face_value, r0, kappa, theta, sigma, T):
    """
    Compare Monte Carlo and analytical bond pricing methods.
    
    Parameters:
    -----------
    face_value : float
        The face value of the zero-coupon bond.
    r0 : float
        Initial short rate.
    kappa : float
        Mean reversion speed.
    theta : float
        Long-term mean level.
    sigma : float
        Volatility of the short rate.
    T : float
        Time to maturity in years.
    """
    # Calculate the bond price using Monte Carlo simulation
    mc_price, _ = monte_carlo_simulation(face_value, r0, kappa, theta, sigma, T)
    
    # Calculate the analytical bond price
    analytical_price = calculate_analytical_price(face_value, r0, kappa, theta, sigma, T)
    
    # Calculate the difference and percentage error
    abs_diff = abs(mc_price - analytical_price)
    pct_error = abs_diff / analytical_price * 100
    
    print("\nComparison of Pricing Methods:")
    print(f"Monte Carlo Price: ${mc_price:.2f}")
    print(f"Analytical Price: ${analytical_price:.2f}")
    print(f"Absolute Difference: ${abs_diff:.2f}")
    print(f"Percentage Error: {pct_error:.4f}%")
    
    # Save comparison results
    comparison_df = pd.DataFrame({
        'method': ['Monte Carlo', 'Analytical'],
        'price': [mc_price, analytical_price],
        'difference': [abs_diff, 0],
        'percentage_error': [pct_error, 0]
    })
    comparison_df.to_csv("./data/vasicekBond/method_comparison.csv", index=False)
    
    # Create a bar chart to visualize the comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    methods = ['Monte Carlo', 'Analytical']
    prices = [mc_price, analytical_price]
    
    bars = ax.bar(methods, prices, width=0.6, color=['skyblue', 'lightgreen'])
    
    # Add price labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'${height:.2f}', ha='center', va='bottom')
    
    # Set labels and title
    ax.set_xlabel('Pricing Method', fontsize=12)
    ax.set_ylabel('Bond Price ($)', fontsize=12)
    ax.set_title('Comparison of Bond Pricing Methods', fontsize=14)
    
    # Add error information in a text box
    textstr = '\n'.join((
        f'Absolute Difference: ${abs_diff:.2f}',
        f'Percentage Error: {pct_error:.4f}%'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.5, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            horizontalalignment='center', bbox=props)
    
    plt.tight_layout()
    plt.savefig('./figs/vasicekBond/method_comparison.png', dpi=300)
    plt.savefig('./figs/vasicekBond/method_comparison.pdf')
    plt.close()
    
    return mc_price, analytical_price

if __name__ == "__main__":
    # Ensure output directories exist
    ensure_directories_exist()
    
    # Set bond and model parameters
    face_value = 1000      # Face value of the bond
    r0 = 0.05              # Initial interest rate (5%)
    kappa = 0.2            # Speed of mean reversion (slower, more realistic)
    theta = 0.06           # Long-term mean level (6%)
    sigma = 0.01           # Volatility (1%, more realistic)
    maturity = 5           # Time to maturity in years (5-year bond)
    
    # Run the simulation
    print("\nRunning Vasicek Bond Price Simulation:")
    print("======================================")
    
    # Compare Monte Carlo and analytical methods
    compare_methods(face_value, r0, kappa, theta, sigma, maturity)
    
    print("\nSimulation completed successfully.")
    print(f"Results saved to ./data/vasicekBond/")
    print(f"Plots saved to ./figs/vasicekBond/")