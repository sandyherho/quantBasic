#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vasicek Bond Pricing Simulation

A simple implementation of the Vasicek short-rate model
to price zero-coupon bonds using Monte Carlo simulation.

Author: Sandy Herho <sandy.herho@email.ucr.edu>
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set plotting style
plt.style.use("bmh")

# Simulation parameters
NUM_OF_SIMULATIONS = 1000  # Number of simulations to run
NUM_OF_POINTS = 200        # Number of time steps per simulation


def ensure_directories():
    """Create output directories if they don't exist."""
    dirs = ["./figs/vasicekBond", "./data/vasicekBond"]
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def monte_carlo_simulation(face_value, r0, kappa, theta, sigma, T=1):
    """
    Monte Carlo simulation of Vasicek model for bond pricing.
    
    Parameters:
    -----------
    face_value : float
        Face value of the zero-coupon bond
    r0 : float
        Initial interest rate
    kappa : float
        Speed of mean reversion
    theta : float
        Long-term mean level
    sigma : float
        Volatility parameter
    T : float
        Time to maturity (in years)
    
    Returns:
    --------
    float
        Calculated bond price
    """
    # Calculate time step
    dt = T / float(NUM_OF_POINTS)
    
    # Initialize storage for simulation results
    result = []
    
    # Run simulations
    print(f"Running {NUM_OF_SIMULATIONS} simulations...")
    for i in range(NUM_OF_SIMULATIONS):
        # Initial rate
        rates = [r0]
        
        # Simulate rate path
        for j in range(NUM_OF_POINTS):
            # Vasicek model: dr = kappa * (theta - r) * dt + sigma * sqrt(dt) * N(0,1)
            dr = kappa * (theta - rates[-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
            rates.append(rates[-1] + dr)
        
        result.append(rates)
    
    # Convert to DataFrame for analysis
    simulation_data = pd.DataFrame(result).T
    
    # Calculate the integral of r(t) based on simulated paths
    integral_sum = simulation_data.sum() * dt
    
    # Present value of the future cash flow
    present_integral_sum = np.exp(-integral_sum)
    
    # Bond price is the mean of discounted values
    bond_price = face_value * np.mean(present_integral_sum)
    
    # Save simulation data
    simulation_data.to_csv("./data/vasicekBond/simulation_data.csv")
    
    return bond_price, simulation_data


def plot_results(simulation_data, bond_price, r0, kappa, theta, sigma, T, face_value):
    """
    Create and save a plot of the simulation results.
    
    Parameters:
    -----------
    simulation_data : pandas.DataFrame
        Dataframe with simulation results
    bond_price : float
        Calculated bond price
    r0, kappa, theta, sigma : float
        Model parameters
    T : float
        Time to maturity
    face_value : float
        Face value of the bond
    """
    # Create time points for x-axis
    time_points = np.linspace(0, T, simulation_data.shape[0])
    
    # Calculate mean rates for each time point
    mean_rates = simulation_data.mean(axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot a sample of 50 paths
    sample_cols = np.random.choice(simulation_data.columns, size=min(50, NUM_OF_SIMULATIONS), replace=False)
    for col in sample_cols:
        ax.plot(time_points, simulation_data[col], color='gray', alpha=0.1, linewidth=0.8)
    
    # Plot mean path
    ax.plot(time_points, mean_rates, 'b-', linewidth=2, label='Mean Rate')
    
    # Plot long-term mean
    ax.axhline(y=theta, color='r', linestyle='--', alpha=0.7, label=f'Long-term Mean (θ)')
    
    # Labels and title
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Interest Rate')
    ax.set_title(f'Vasicek Model Simulation\nBond Price: ${bond_price:.2f}')
    
    # Add legend
    ax.legend()
    
    # Add text box with parameters
    textstr = '\n'.join((
        f'Parameters:',
        f'r₀ = {r0:.2f}',
        f'κ = {kappa:.2f}',
        f'θ = {theta:.2f}',
        f'σ = {sigma:.2f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('./figs/vasicekBond/vasicek_simulation.png', dpi=300)
    plt.close()


def main():
    """Main function to run the simulation and generate plot."""
    # Ensure directories exist
    ensure_directories()
    
    # Set model parameters
    face_value = 1000
    r0 = 0.1        # Initial interest rate
    kappa = 0.3     # Speed of mean reversion
    theta = 0.3     # Long-term mean level
    sigma = 0.03    # Volatility
    T = 1           # Time to maturity in years
    
    # Run simulation
    bond_price, simulation_data = monte_carlo_simulation(face_value, r0, kappa, theta, sigma, T)
    
    # Print results
    print(f"Bond price based on Monte-Carlo simulation: ${bond_price:.2f}")
    
    # Plot results
    plot_results(simulation_data, bond_price, r0, kappa, theta, sigma, T, face_value)
    
    print("Simulation complete. Results saved to ./data/vasicekBond/ and ./figs/vasicekBond/")


if __name__ == "__main__":
    main()