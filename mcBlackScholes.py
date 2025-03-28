#!/usr/bin/env python
"""
Monte Carlo Simulation for European Option Pricing
This program calculates call and put option prices using Monte Carlo simulation
and visualizes the simulated stock price paths.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
plt.style.use("bmh")

class OptionPricing:
    def __init__(self, S0, E, T, rf, sigma, iterations):
        """
        Initialize the OptionPricing model with given parameters.
        
        Parameters:
        S0 (float): Initial stock price
        E (float): Strike price
        T (float): Time to maturity in years
        rf (float): Risk-free interest rate
        sigma (float): Volatility
        iterations (int): Number of Monte Carlo simulations
        """
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations
        
        # Create figures directory if it doesn't exist
        if not os.path.exists('./figs'):
            os.makedirs('./figs')

    def call_option_simulation(self):
        """
        Calculate the price of a European call option using Monte Carlo simulation.
        
        Returns:
        float: Estimated call option price
        """
        # Initialize option data array with two columns:
        # Column 1: zeros (for comparison in max function)
        # Column 2: will store the payoff values
        option_data = np.zeros([self.iterations, 2])
        
        # Generate random normal variables for the simulation
        rand = np.random.normal(0, 1, [1, self.iterations])

        # Calculate stock price at maturity using GBM formula
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma**2) +
                                     self.sigma * np.sqrt(self.T) * rand)

        # Calculate payoff: max(S-E, 0) for call option
        option_data[:, 1] = stock_price - self.E

        # Calculate average payoff across all simulations
        average = np.sum(np.amax(option_data, axis=1)) / float(self.iterations)
        
        # Discount the average payoff to present value
        return np.exp(-1 * self.rf * self.T) * average

    def put_option_simulation(self):
        """
        Calculate the price of a European put option using Monte Carlo simulation.
        
        Returns:
        float: Estimated put option price
        """
        # Initialize option data array with two columns:
        # Column 1: zeros (for comparison in max function)
        # Column 2: will store the payoff values
        option_data = np.zeros([self.iterations, 2])
        
        # Generate random normal variables for the simulation
        rand = np.random.normal(0, 1, [1, self.iterations])

        # Calculate stock price at maturity using GBM formula
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma**2) +
                                     self.sigma * np.sqrt(self.T) * rand)

        # Calculate payoff: max(E-S, 0) for put option
        option_data[:, 1] = self.E - stock_price

        # Calculate average payoff across all simulations
        average = np.sum(np.amax(option_data, axis=1)) / float(self.iterations)
        
        # Discount the average payoff to present value
        return np.exp(-1 * self.rf * self.T) * average
    
    def plot_simulated_prices(self, n_paths=1000):
        """
        Plot simulated stock price paths and save to ./figs/
        
        Parameters:
        n_paths (int): Number of paths to plot (for clarity)
        """
        if n_paths > self.iterations:
            n_paths = self.iterations
        
        # Generate time points
        time_points = np.linspace(0, self.T, 100)
        
        # Initialize array for price paths
        price_paths = np.zeros((n_paths, len(time_points)))
        
        # Generate random numbers for simulation
        rand = np.random.normal(0, 1, [n_paths, len(time_points)])
        
        # Calculate price paths
        for i, t in enumerate(time_points):
            if i == 0:
                price_paths[:, i] = self.S0
            else:
                dt = time_points[i] - time_points[i-1]
                price_paths[:, i] = price_paths[:, i-1] * np.exp(
                    (self.rf - 0.5 * self.sigma**2) * dt +
                    self.sigma * np.sqrt(dt) * rand[:, i]
                )
        
        # Plot the simulated paths
        plt.figure(figsize=(12, 6))
        for path in price_paths:
            plt.plot(time_points, path, lw=1, alpha=0.5)
        
        plt.axhline(y=self.E, color='r', linestyle='--', label='Strike Price')
        plt.title(f'Simulated Stock Price Paths (n={n_paths})')
        plt.xlabel('Time to Maturity (Years)')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True)
        
        # Save the figure
        plt.savefig('./figs/blackScholesMC_simulated_paths.png')
        plt.close()

if __name__ == "__main__":
    # Initialize the model with parameters
    model = OptionPricing(100, 100, 1, 0.05, 0.2, 1000000)
    
    # Calculate and print option prices
    call_price = model.call_option_simulation()
    put_price = model.put_option_simulation()
    print("The value of the call option: $%.2f" % call_price)
    print("The value of the put option: $%.2f" % put_price)
    
    # Plot and save simulated paths
    model.plot_simulated_prices(n_paths=100)