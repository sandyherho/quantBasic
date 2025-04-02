#!/usr/bin/env python
"""
Black-Scholes Option Pricing Model

This script implements the Black-Scholes model for pricing European call and put options.
The model assumes constant volatility, no dividends, and a constant risk-free rate.

References:
- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.
  Journal of Political Economy, 81(3), 637-654.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Set the style for plots
plt.style.use("bmh")

def calculate_d_parameters(S, E, rf, sigma, T, t=0):
    """
    Calculate the d1 and d2 parameters used in the Black-Scholes formula.
    
    Parameters:
    -----------
    S : float
        Current stock price
    E : float
        Strike price
    rf : float
        Risk-free interest rate (annual)
    sigma : float
        Volatility of the underlying asset (annual)
    T : float
        Time to expiration (in years)
    t : float, optional
        Current time (default is 0)
        
    Returns:
    --------
    tuple
        (d1, d2) parameters
    """
    # Check for valid time to expiration
    time_to_expiration = T - t
    if time_to_expiration <= 0:
        raise ValueError("Time to expiration must be positive")
    
    # Calculate d1 parameter
    d1 = (np.log(S/E) + (rf + (sigma**2)/2) * time_to_expiration) / (sigma * np.sqrt(time_to_expiration))
    
    # Calculate d2 parameter
    d2 = d1 - sigma * np.sqrt(time_to_expiration)
    
    return d1, d2

def call_option_price(S, E, rf, sigma, T, t=0, verbose=True):
    """
    Calculate the price of a European call option using the Black-Scholes model.
    
    Parameters:
    -----------
    S : float
        Current stock price
    E : float
        Strike price
    rf : float
        Risk-free interest rate (annual)
    sigma : float
        Volatility of the underlying asset (annual)
    T : float
        Time to expiration (in years)
    t : float, optional
        Current time (default is 0)
    verbose : bool, optional
        Whether to print intermediate calculations (default is True)
        
    Returns:
    --------
    float
        Price of the call option
    """
    # Calculate d1 and d2 parameters
    d1, d2 = calculate_d_parameters(S, E, rf, sigma, T, t)
    
    if verbose:
        print(f"The d1 & d2 parameters: ({d1:.6f}, {d2:.6f})")
    
    # Calculate call option price using the Black-Scholes formula
    # Call = S * N(d1) - E * e^(-rf * (T-t)) * N(d2)
    call_price = S * stats.norm.cdf(d1) - E * np.exp(-rf * (T-t)) * stats.norm.cdf(d2)
    
    return call_price

def put_option_price(S, E, rf, sigma, T, t=0, verbose=True):
    """
    Calculate the price of a European put option using the Black-Scholes model.
    
    Parameters:
    -----------
    S : float
        Current stock price
    E : float
        Strike price
    rf : float
        Risk-free interest rate (annual)
    sigma : float
        Volatility of the underlying asset (annual)
    T : float
        Time to expiration (in years)
    t : float, optional
        Current time (default is 0)
    verbose : bool, optional
        Whether to print intermediate calculations (default is True)
        
    Returns:
    --------
    float
        Price of the put option
    """
    # Calculate d1 and d2 parameters
    d1, d2 = calculate_d_parameters(S, E, rf, sigma, T, t)
    
    if verbose:
        print(f"The d1 & d2 parameters: ({d1:.6f}, {d2:.6f})")
    
    # Calculate put option price using the Black-Scholes formula
    # Put = E * e^(-rf * (T-t)) * N(-d2) - S * N(-d1)
    put_price = E * np.exp(-rf * (T-t)) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    
    return put_price

def put_call_parity(call_price, put_price, S, E, rf, T, t=0):
    """
    Verify the put-call parity relationship.
    Put-call parity: C + Ke^(-r(T-t)) = P + S
    
    Parameters:
    -----------
    call_price : float
        Price of the call option
    put_price : float
        Price of the put option
    S : float
        Current stock price
    E : float
        Strike price
    rf : float
        Risk-free interest rate (annual)
    T : float
        Time to expiration (in years)
    t : float, optional
        Current time (default is 0)
        
    Returns:
    --------
    tuple
        (left_side, right_side, difference) of the put-call parity equation
    """
    left_side = call_price + E * np.exp(-rf * (T-t))
    right_side = put_price + S
    difference = abs(left_side - right_side)
    
    return left_side, right_side, difference

def plot_option_values(S_range, E, rf, sigma, T, t=0):
    """
    Plot call and put option values for a range of underlying prices.
    
    Parameters:
    -----------
    S_range : array-like
        Range of stock prices to evaluate
    E : float
        Strike price
    rf : float
        Risk-free interest rate (annual)
    sigma : float
        Volatility of the underlying asset (annual)
    T : float
        Time to expiration (in years)
    t : float, optional
        Current time (default is 0)
    """
    call_prices = []
    put_prices = []
    
    for S in S_range:
        call_prices.append(call_option_price(S, E, rf, sigma, T, t, verbose=False))
        put_prices.append(put_option_price(S, E, rf, sigma, T, t, verbose=False))
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(S_range, call_prices, 'b-', linewidth=2)
    plt.axvline(x=E, color='gray', linestyle='--')
    plt.grid(True)
    plt.title('Call Option Value')
    plt.xlabel('Stock Price')
    plt.ylabel('Option Value')
    
    plt.subplot(1, 2, 2)
    plt.plot(S_range, put_prices, 'r-', linewidth=2)
    plt.axvline(x=E, color='gray', linestyle='--')
    plt.grid(True)
    plt.title('Put Option Value')
    plt.xlabel('Stock Price')
    plt.ylabel('Option Value')
    
    plt.tight_layout()
    plt.show()

def calculate_greeks(S, E, rf, sigma, T, t=0):
    """
    Calculate the option Greeks (Delta, Gamma, Theta, Vega, Rho).
    
    Parameters:
    -----------
    S : float
        Current stock price
    E : float
        Strike price
    rf : float
        Risk-free interest rate (annual)
    sigma : float
        Volatility of the underlying asset (annual)
    T : float
        Time to expiration (in years)
    t : float, optional
        Current time (default is 0)
        
    Returns:
    --------
    dict
        Dictionary containing the Greeks for both call and put options
    """
    d1, d2 = calculate_d_parameters(S, E, rf, sigma, T, t)
    time_to_expiration = T - t
    
    # Calculate probability density function of d1
    d1_pdf = stats.norm.pdf(d1)
    
    # Calculate Greeks
    call_delta = stats.norm.cdf(d1)
    put_delta = call_delta - 1
    
    gamma = d1_pdf / (S * sigma * np.sqrt(time_to_expiration))
    
    call_theta = -((S * sigma * d1_pdf) / (2 * np.sqrt(time_to_expiration))) - rf * E * np.exp(-rf * time_to_expiration) * stats.norm.cdf(d2)
    put_theta = -((S * sigma * d1_pdf) / (2 * np.sqrt(time_to_expiration))) + rf * E * np.exp(-rf * time_to_expiration) * stats.norm.cdf(-d2)
    
    vega = S * np.sqrt(time_to_expiration) * d1_pdf
    
    call_rho = E * time_to_expiration * np.exp(-rf * time_to_expiration) * stats.norm.cdf(d2)
    put_rho = -E * time_to_expiration * np.exp(-rf * time_to_expiration) * stats.norm.cdf(-d2)
    
    # Return Greeks as a dictionary
    return {
        'call_delta': call_delta,
        'put_delta': put_delta,
        'gamma': gamma,
        'call_theta': call_theta,
        'put_theta': put_theta,
        'vega': vega,
        'call_rho': call_rho,
        'put_rho': put_rho
    }

if __name__ == '__main__':
    # Example usage
    
    # Parameters
    S0 = 100       # Current stock price
    E = 100        # Strike price
    T = 1          # Time to expiration (1 year)
    rf = 0.05      # Risk-free rate (5%)
    sigma = 0.2    # Volatility (20%)
    
    # Calculate option prices
    call_price = call_option_price(S0, E, rf, sigma, T)
    put_price = put_option_price(S0, E, rf, sigma, T)
    
    print("\nCall option price according to Black-Scholes model: {:.4f}".format(call_price))
    print("Put option price according to Black-Scholes model: {:.4f}".format(put_price))
    
    # Verify put-call parity
    left, right, diff = put_call_parity(call_price, put_price, S0, E, rf, T)
    print("\nPut-Call Parity Check:")
    print("Left side: {:.4f}".format(left))
    print("Right side: {:.4f}".format(right))
    print("Difference: {:.8f}".format(diff))
    
    # Calculate Greeks
    greeks = calculate_greeks(S0, E, rf, sigma, T)
    print("\nOption Greeks:")
    for greek, value in greeks.items():
        print(f"{greek}: {value:.6f}")
    
    # Plot option values for a range of stock prices
    S_range = np.linspace(50, 150, 100)
    plot_option_values(S_range, E, rf, sigma, T)
