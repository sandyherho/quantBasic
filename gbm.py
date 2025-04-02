#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Geometric Brownian Motion Simulation

This script simulates and plots a geometric Brownian motion
which is commonly used for stock price modeling.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set plotting style
plt.style.use("bmh")


def simulate_geometric_random_walk(S0, T=2, N=10000, mu=0.1, sigma=0.05):
    """
    Simulate a geometric Brownian motion.
    
    Parameters:
    -----------
    S0 : float
        Initial stock price
    T : float
        Total time period for simulation
    N : int
        Number of time steps
    mu : float
        Drift coefficient (expected return)
    sigma : float
        Volatility (standard deviation of returns)
        
    Returns:
    --------
    t : numpy.ndarray
        Time points
    S : numpy.ndarray
        Simulated stock prices
    """
    # Calculate time step
    dt = T / N
    
    # Create time array
    t = np.linspace(0, T, N)
    
    # Generate random increments from standard normal distribution N(0,1)
    W = np.random.standard_normal(size=N)
    
    # Scale to N(0, dt) and create Brownian motion by cumulative sum
    W = np.cumsum(W) * np.sqrt(dt)
    
    # Calculate log returns
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    
    # Calculate stock price path
    S = S0 * np.exp(X)
    
    return t, S


def plot_simulation(t, S, save_path=None):
    """
    Plot the simulated stock price path.
    
    Parameters:
    -----------
    t : numpy.ndarray
        Time points
    S : numpy.ndarray
        Stock prices
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, S)
    plt.xlabel('Time $t$')
    plt.ylabel('Stock Price $S(t)$')
    plt.title("Geometric Brownian Motion")
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show()


def ensure_dir(directory):
    """
    Create directory if it doesn't exist.
    
    Parameters:
    -----------
    directory : str
        Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


if __name__ == '__main__':
    # Define save directory
    save_dir = "./figs/randomBehave/"
    
    # Create directory if it doesn't exist
    ensure_dir(save_dir)
    
    # Define save path for the figure
    fig_path = os.path.join(save_dir, "geometric_brownian_motion.png")
    
    # Run simulation
    time, data = simulate_geometric_random_walk(S0=10)
    
    # Plot and save results
    plot_simulation(time, data, save_path=fig_path)