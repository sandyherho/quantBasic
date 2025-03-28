#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os

# Set plotting style
plt.style.use("bmh")

def vasicek_model(r0, kappa, theta, sigma, T=1, N=1000):
    """
    Simulate interest rates using the Vasicek model.
    
    Parameters:
    -----------
    r0 : float
        Initial interest rate
    kappa : float
        Mean reversion speed
    theta : float
        Long-term mean interest rate
    sigma : float
        Volatility of interest rate
    T : float
        Time horizon
    N : int
        Number of time steps
        
    Returns:
    --------
    t : numpy.ndarray
        Time points
    rates : list
        Simulated interest rates at each time point
    """
    dt = T/float(N)
    t = np.linspace(0, T, N+1)
    rates = [r0]
    
    for _ in range(N):
        dr = kappa * (theta - rates[-1]) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1)
        rates.append(rates[-1] + dr)
    return t, rates

def plot_model(t, r, save_path=None):
    """
    Plot the Vasicek model simulation and optionally save the figure.
    
    Parameters:
    -----------
    t : numpy.ndarray
        Time points
    r : list
        Interest rates at each time point
    save_path : str, optional
        Path where the figure should be saved. If None, figure is not saved.
    """
    plt.figure(figsize=(15, 6))
    plt.plot(t, r)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$r(t)$")
    plt.title("Vasicek Interest Rate Model")
    
    # Save figure if a path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
              
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define the save path
    save_path = "./figs/ornsteinUhlenbeck/simpleVasicek.png"
    
    # Check if the directory exists, if not create it
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        print(f"Created directory: {os.path.dirname(save_path)}")
    
    # Run the Vasicek model simulation with typical parameters
    # r0: initial interest rate (0.05 = 5%)
    # kappa: mean reversion speed (0.3 typical for short-term rates)
    # theta: long-term mean rate (0.05 = 5%)
    # sigma: volatility (0.02 = 2% typical for interest rates)
    # T: time horizon in years (5 years)
    time, data = vasicek_model(r0=0.05, kappa=0.3, theta=0.05, sigma=0.02, T=5, N=1000)
    
    # Plot and save the results
    plot_model(time, data, save_path)