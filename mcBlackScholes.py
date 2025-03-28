#!/usr/bin/env python
"""
Monte Carlo Simulation for European Option Pricing
This program calculates call and put option prices using Monte Carlo simulation.
"""

import numpy as np

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

        # Calculate stock price at maturity using Geometric Brownian Motion formula
        stock_price = self.S0 * np.exp(
            self.T * (self.rf - 0.5 * self.sigma**2) + 
            self.sigma * np.sqrt(self.T) * rand
        )

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

        # Calculate stock price at maturity using Geometric Brownian Motion formula
        stock_price = self.S0 * np.exp(
            self.T * (self.rf - 0.5 * self.sigma**2) + 
            self.sigma * np.sqrt(self.T) * rand
        )

        # Calculate payoff: max(E-S, 0) for put option
        option_data[:, 1] = self.E - stock_price

        # Calculate average payoff across all simulations
        average = np.sum(np.amax(option_data, axis=1)) / float(self.iterations)
        
        # Discount the average payoff to present value
        return np.exp(-1 * self.rf * self.T) * average

if __name__ == "__main__":
    # Initialize the model with parameters
    model = OptionPricing(
        S0=100,         # Initial stock price
        E=100,          # Strike price
        T=1.0,          # Time to maturity (1 year)
        rf=0.05,        # Risk-free rate (5%)
        sigma=0.2,      # Volatility (20%)
        iterations=1000000  # Number of simulations
    )
    
    # Calculate and print option prices
    print("The value of the call option: $%.2f" % model.call_option_simulation())
    print("The value of the put option: $%.2f" % model.put_option_simulation())