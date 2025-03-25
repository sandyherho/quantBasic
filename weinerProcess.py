#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script simulates and visualizes a Wiener process (also known as Brownian motion).
It creates a random path based on normal distribution increments and saves the plot
to a specified directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Set plot style
plt.style.use("bmh")

def weiner_process(dt=0.1, x0=0, n=1000, seed=None):
    """
    Generate a Wiener process (Brownian motion) path.
    
    Parameters:
    -----------
    dt : float
        Time step size
    x0 : float
        Initial position
    n : int
        Number of steps to simulate
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    t : numpy.ndarray
        Time points
    W : numpy.ndarray
        Values of the Wiener process at each time point
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize array with zeros (W(t=0) = 0)
    W = np.zeros(n+1)
    
    # Create time steps from x0 to n
    t = np.linspace(x0, n*dt, n+1)
    
    # Generate random increments and compute cumulative sum
    # Each increment is normally distributed with mean 0 and variance dt
    W[1:n+1] = np.cumsum(np.random.normal(0, np.sqrt(dt), n))
    
    return t, W

def plot_process(t, W, save_path=None, show=True):
    """
    Plot the Wiener process and optionally save the figure.
    
    Parameters:
    -----------
    t : numpy.ndarray
        Time points
    W : numpy.ndarray
        Values of the Wiener process at each time point
    save_path : str, optional
        Path to save the figure
    show : bool, optional
        Whether to display the plot
    """
    # Create figure and plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, W)
    
    # Add labels and title
    plt.xlabel('Time - $t$')
    plt.ylabel('Wiener Process - $W(t)$')
    plt.title('Wiener Process Simulation')
    
    # Add timestamp to plot
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.figtext(0.02, 0.02, f'Generated: {timestamp}', fontsize=8)
    
    # Save figure if path is provided
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    # Show plot if requested
    if show:
        plt.show()
    else:
        plt.close()

def run_simulation(dt=0.1, x0=0, n=10000, num_simulations=1, save_dir='./figs/randomBehave'):
    """
    Run multiple simulations of the Wiener process and save the results.
    
    Parameters:
    -----------
    dt : float
        Time step size
    x0 : float
        Initial position
    n : int
        Number of steps to simulate
    num_simulations : int
        Number of simulations to run
    save_dir : str
        Directory to save the figures
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(num_simulations):
        # Generate a unique filename based on parameters and simulation number
        filename = f"wiener_dt{dt}_n{n}_sim{i+1}.png"
        save_path = os.path.join(save_dir, filename)
        
        # Run simulation
        time, data = weiner_process(dt=dt, x0=x0, n=n)
        
        # Plot and save results
        plot_process(time, data, save_path=save_path, show=(i == num_simulations-1))
        
        print(f"Completed simulation {i+1}/{num_simulations}")

if __name__ == '__main__':
    # Default parameters
    params = {
        'dt': 0.1,         # Time step
        'x0': 0,           # Initial position
        'n': 10000,        # Number of steps
        'num_sims': 1      # Number of simulations
    }
    
    # Run the simulation with default parameters
    # Results will be saved to ./figs/randomBehave
    run_simulation(
        dt=params['dt'],
        x0=params['x0'],
        n=params['n'],
        num_simulations=params['num_sims']
    )