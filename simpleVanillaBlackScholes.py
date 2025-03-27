#!/usr/bin/env python
"""
Black-Scholes Option Pricing Model

This script implements the Black-Scholes model for pricing European call and put options.
The model assumes constant volatility, no dividends, and a constant risk-free rate.

The script organizes data outputs in './data/simpleBlackScholes/' and
figure outputs in './figs/simpleBlackScholes/'.

References:
- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.
  Journal of Political Economy, 81(3), 637-654.
- Hull, J. C. (2018). Options, Futures, and Other Derivatives (10th ed.). Pearson.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import datetime
import json
from matplotlib.ticker import MultipleLocator
from matplotlib import rcParams

# Configure matplotlib for publication-quality figures
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 10
rcParams['figure.figsize'] = (8, 6)
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['lines.linewidth'] = 1.5
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.1
plt.style.use("bmh")

# Create necessary directories
def ensure_directories_exist():
    """
    Create the necessary directories for data and figures if they don't exist.
    """
    data_dir = os.path.join(".", "data", "simpleBlackScholes")
    figs_dir = os.path.join(".", "figs", "simpleBlackScholes")
    
    for directory in [data_dir, figs_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    return data_dir, figs_dir

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
    
    # Theta is expressed as the option price change per calendar day
    # Dividing by 365 converts annual rate to daily
    call_theta = (-((S * sigma * d1_pdf) / (2 * np.sqrt(time_to_expiration))) - 
                  rf * E * np.exp(-rf * time_to_expiration) * stats.norm.cdf(d2)) / 365
    
    put_theta = (-((S * sigma * d1_pdf) / (2 * np.sqrt(time_to_expiration))) + 
                rf * E * np.exp(-rf * time_to_expiration) * stats.norm.cdf(-d2)) / 365
    
    # Vega is expressed as the option price change per 1% change in volatility
    # Multiplying by 0.01 scales it to a 1% change
    vega = S * np.sqrt(time_to_expiration) * d1_pdf * 0.01
    
    # Rho is expressed as the option price change per 1% change in interest rate
    # Multiplying by 0.01 scales it to a 1% change
    call_rho = E * time_to_expiration * np.exp(-rf * time_to_expiration) * stats.norm.cdf(d2) * 0.01
    put_rho = -E * time_to_expiration * np.exp(-rf * time_to_expiration) * stats.norm.cdf(-d2) * 0.01
    
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

def implied_volatility(option_price, S, E, rf, T, t=0, option_type='call', precision=0.0001, max_iterations=100):
    """
    Calculate implied volatility using the Newton-Raphson method.
    
    Parameters:
    -----------
    option_price : float
        Market price of the option
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
    option_type : str, optional
        Type of option ('call' or 'put', default is 'call')
    precision : float, optional
        Desired precision (default is 0.0001)
    max_iterations : int, optional
        Maximum number of iterations (default is 100)
        
    Returns:
    --------
    float
        Implied volatility
    """
    # Initial guess for implied volatility
    sigma = 0.2
    
    for i in range(max_iterations):
        if option_type.lower() == 'call':
            price = call_option_price(S, E, rf, sigma, T, t, verbose=False)
            vega = calculate_greeks(S, E, rf, sigma, T, t)['vega'] * 100  # Adjust vega scaling
        else:
            price = put_option_price(S, E, rf, sigma, T, t, verbose=False)
            vega = calculate_greeks(S, E, rf, sigma, T, t)['vega'] * 100  # Adjust vega scaling
        
        price_diff = option_price - price
        
        if abs(price_diff) < precision:
            return sigma
        
        # Update sigma using Newton-Raphson
        sigma = sigma + price_diff / vega
        
        # Ensure sigma remains positive
        if sigma <= 0:
            sigma = 0.001
    
    print(f"Warning: Implied volatility calculation did not converge after {max_iterations} iterations.")
    return sigma

def plot_option_values(S_range, E, rf, sigma, T, t=0, save_path=None):
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
    save_path : str, optional
        Path to save the figure (default is None, which doesn't save)
    """
    call_prices = []
    put_prices = []
    intrinsic_call = []
    intrinsic_put = []
    
    for S in S_range:
        call_prices.append(call_option_price(S, E, rf, sigma, T, t, verbose=False))
        put_prices.append(put_option_price(S, E, rf, sigma, T, t, verbose=False))
        intrinsic_call.append(max(0, S - E))
        intrinsic_put.append(max(0, E - S))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Call option plot
    ax1.plot(S_range, call_prices, 'b-', linewidth=2, label='Call Option Value')
    ax1.plot(S_range, intrinsic_call, 'g--', linewidth=1.5, label='Intrinsic Value')
    ax1.axvline(x=E, color='gray', linestyle='--', label='Strike Price')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Call Option Value', fontweight='bold')
    ax1.set_xlabel('Stock Price ($)')
    ax1.set_ylabel('Option Value ($)')
    ax1.legend()
    ax1.xaxis.set_major_locator(MultipleLocator(10))
    
    # Put option plot
    ax2.plot(S_range, put_prices, 'r-', linewidth=2, label='Put Option Value')
    ax2.plot(S_range, intrinsic_put, 'g--', linewidth=1.5, label='Intrinsic Value')
    ax2.axvline(x=E, color='gray', linestyle='--', label='Strike Price')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Put Option Value', fontweight='bold')
    ax2.set_xlabel('Stock Price ($)')
    ax2.set_ylabel('Option Value ($)')
    ax2.legend()
    ax2.xaxis.set_major_locator(MultipleLocator(10))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

def plot_greeks(S_range, E, rf, sigma, T, t=0, save_dir=None):
    """
    Plot option Greeks for a range of underlying prices.
    
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
    save_dir : str, optional
        Directory to save the figures (default is None, which doesn't save)
    """
    # Initialize lists to store Greek values
    call_delta = []
    put_delta = []
    gamma = []
    call_theta = []
    put_theta = []
    vega = []
    call_rho = []
    put_rho = []
    
    # Calculate Greeks for each stock price
    for S in S_range:
        greeks = calculate_greeks(S, E, rf, sigma, T, t)
        call_delta.append(greeks['call_delta'])
        put_delta.append(greeks['put_delta'])
        gamma.append(greeks['gamma'])
        call_theta.append(greeks['call_theta'])
        put_theta.append(greeks['put_theta'])
        vega.append(greeks['vega'])
        call_rho.append(greeks['call_rho'])
        put_rho.append(greeks['put_rho'])
    
    # Create and save plots for each Greek
    greek_plots = [
        ('Delta', [call_delta, put_delta], ['Call Delta', 'Put Delta'], ['blue', 'red']),
        ('Gamma', [gamma], ['Gamma'], ['green']),
        ('Theta', [call_theta, put_theta], ['Call Theta', 'Put Theta'], ['blue', 'red']),
        ('Vega', [vega], ['Vega'], ['purple']),
        ('Rho', [call_rho, put_rho], ['Call Rho', 'Put Rho'], ['blue', 'red'])
    ]
    
    for greek_name, data_list, labels, colors in greek_plots:
        plt.figure(figsize=(8, 6))
        
        for data, label, color in zip(data_list, labels, colors):
            plt.plot(S_range, data, color=color, linewidth=2, label=label)
        
        plt.axvline(x=E, color='gray', linestyle='--', label='Strike Price')
        plt.grid(True, alpha=0.3)
        plt.title(f'{greek_name} vs. Stock Price', fontweight='bold')
        plt.xlabel('Stock Price ($)')
        plt.ylabel(greek_name)
        plt.legend()
        
        if save_dir:
            save_path = os.path.join(save_dir, f'greek_{greek_name.lower()}.png')
            plt.savefig(save_path)
            print(f"Figure saved to: {save_path}")
            plt.close()
        else:
            plt.show()

def plot_iv_surface(strike_range, expiry_range, S, rf, call_prices=None, save_path=None):
    """
    Plot implied volatility surface.
    
    Parameters:
    -----------
    strike_range : array-like
        Range of strike prices
    expiry_range : array-like
        Range of expiry times (in years)
    S : float
        Current stock price
    rf : float
        Risk-free interest rate (annual)
    call_prices : callable, optional
        Function to calculate call prices (default is None, which uses theoretical prices)
    save_path : str, optional
        Path to save the figure (default is None, which doesn't save)
    """
    # Create meshgrid for strike and expiry
    K, T = np.meshgrid(strike_range, expiry_range)
    
    # Initialize implied volatility surface
    IV = np.zeros_like(K, dtype=float)
    
    # Base volatility used if call_prices is None
    base_sigma = 0.2
    
    # Calculate implied volatility for each point
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            strike = K[i, j]
            expiry = T[i, j]
            
            # Calculate option price based on provided function or theoretical price
            if call_prices is not None:
                price = call_prices(strike, expiry)
            else:
                # Use theoretical price with a volatility smile
                moneyness = S / strike
                adjusted_sigma = base_sigma * (1 + 0.1 * (moneyness - 1)**2)
                price = call_option_price(S, strike, rf, adjusted_sigma, expiry, verbose=False)
            
            # Calculate implied volatility
            try:
                IV[i, j] = implied_volatility(price, S, strike, rf, expiry, option_type='call')
            except:
                IV[i, j] = np.nan
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(K, T, IV, cmap='viridis', alpha=0.8, 
                          linewidth=0, antialiased=True)
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Implied Volatility')
    
    # Set labels and title
    ax.set_xlabel('Strike Price ($)')
    ax.set_ylabel('Time to Expiry (years)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title('Implied Volatility Surface', fontweight='bold')
    
    # Adjust viewing angle
    ax.view_init(elev=30, azim=45)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

def generate_sample_data(S0, E, rf, sigma, T, S_range=None, save_path=None):
    """
    Generate and save sample option data.
    
    Parameters:
    -----------
    S0 : float
        Current stock price
    E : float
        Strike price
    rf : float
        Risk-free interest rate (annual)
    sigma : float
        Volatility of the underlying asset (annual)
    T : float
        Time to expiration (in years)
    S_range : array-like, optional
        Range of stock prices (default is None, which uses a standard range)
    save_path : str, optional
        Path to save the data (default is None, which doesn't save)
    
    Returns:
    --------
    pandas.DataFrame
        Generated sample data
    """
    if S_range is None:
        S_range = np.linspace(0.7 * S0, 1.3 * S0, 20)
    
    data = []
    
    for S in S_range:
        call_price = call_option_price(S, E, rf, sigma, T, verbose=False)
        put_price = put_option_price(S, E, rf, sigma, T, verbose=False)
        greeks = calculate_greeks(S, E, rf, sigma, T)
        
        data.append({
            'stock_price': S,
            'strike_price': E,
            'risk_free_rate': rf,
            'volatility': sigma,
            'time_to_expiry': T,
            'call_price': call_price,
            'put_price': put_price,
            'call_delta': greeks['call_delta'],
            'put_delta': greeks['put_delta'],
            'gamma': greeks['gamma'],
            'call_theta': greeks['call_theta'],
            'put_theta': greeks['put_theta'],
            'vega': greeks['vega'],
            'call_rho': greeks['call_rho'],
            'put_rho': greeks['put_rho']
        })
    
    df = pd.DataFrame(data)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Data saved to: {save_path}")
    
    return df

def sensitivity_analysis(S0, E, rf, sigma, T, param_ranges, save_dir=None):
    """
    Perform sensitivity analysis for the Black-Scholes model.
    
    Parameters:
    -----------
    S0 : float
        Base stock price
    E : float
        Base strike price
    rf : float
        Base risk-free interest rate (annual)
    sigma : float
        Base volatility of the underlying asset (annual)
    T : float
        Base time to expiration (in years)
    param_ranges : dict
        Dictionary of parameter ranges for sensitivity analysis
    save_dir : str, optional
        Directory to save the figures (default is None, which doesn't save)
    """
    # Parameters to analyze
    params = {
        'stock_price': {'base': S0, 'label': 'Stock Price ($)', 'range': param_ranges.get('stock_price', np.linspace(0.7 * S0, 1.3 * S0, 50))},
        'strike_price': {'base': E, 'label': 'Strike Price ($)', 'range': param_ranges.get('strike_price', np.linspace(0.7 * E, 1.3 * E, 50))},
        'risk_free_rate': {'base': rf, 'label': 'Risk-Free Rate', 'range': param_ranges.get('risk_free_rate', np.linspace(0, 0.1, 50))},
        'volatility': {'base': sigma, 'label': 'Volatility', 'range': param_ranges.get('volatility', np.linspace(0.05, 0.5, 50))},
        'time_to_expiry': {'base': T, 'label': 'Time to Expiry (years)', 'range': param_ranges.get('time_to_expiry', np.linspace(0.1, 2, 50))}
    }
    
    for param_name, param_info in params.items():
        call_prices = []
        put_prices = []
        param_range = param_info['range']
        
        for param_value in param_range:
            # Set parameters based on which one we're varying
            if param_name == 'stock_price':
                S, strike, rate, vol, expiry = param_value, E, rf, sigma, T
            elif param_name == 'strike_price':
                S, strike, rate, vol, expiry = S0, param_value, rf, sigma, T
            elif param_name == 'risk_free_rate':
                S, strike, rate, vol, expiry = S0, E, param_value, sigma, T
            elif param_name == 'volatility':
                S, strike, rate, vol, expiry = S0, E, rf, param_value, T
            else:  # time_to_expiry
                S, strike, rate, vol, expiry = S0, E, rf, sigma, param_value
            
            # Calculate option prices
            call = call_option_price(S, strike, rate, vol, expiry, verbose=False)
            put = put_option_price(S, strike, rate, vol, expiry, verbose=False)
            
            call_prices.append(call)
            put_prices.append(put)
        
        # Plot sensitivity
        plt.figure(figsize=(8, 6))
        plt.plot(param_range, call_prices, 'b-', linewidth=2, label='Call Option')
        plt.plot(param_range, put_prices, 'r-', linewidth=2, label='Put Option')
        
        plt.grid(True, alpha=0.3)
        plt.title(f'Option Price Sensitivity to {param_info["label"]}', fontweight='bold')
        plt.xlabel(param_info['label'])
        plt.ylabel('Option Price ($)')
        plt.legend()
        
        if save_dir:
            save_path = os.path.join(save_dir, f'sensitivity_{param_name}.png')
            plt.savefig(save_path)
            print(f"Figure saved to: {save_path}")
            plt.close()
        else:
            plt.show()

def save_model_parameters(params, save_path):
    """
    Save model parameters to a JSON file.
    
    Parameters:
    -----------
    params : dict
        Dictionary of model parameters
    save_path : str
        Path to save the parameters
    """
    with open(save_path, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"Parameters saved to: {save_path}")

if __name__ == '__main__':
    # Create necessary directories
    data_dir, figs_dir = ensure_directories_exist()
    
    # Parameters
    S0 = 100       # Current stock price
    E = 100        # Strike price
    T = 1          # Time to expiration (1 year)
    rf = 0.05      # Risk-free rate (5%)
    sigma = 0.2    # Volatility (20%)
    
    # Save parameters
    params = {
        'stock_price': S0,
        'strike_price': E,
        'time_to_expiry': T,
        'risk_free_rate': rf,
        'volatility': sigma,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    params_path = os.path.join(data_dir, 'model_parameters.json')
    save_model_parameters(params, params_path)
    
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
    
    # Generate and save sample data
    S_range = np.linspace(50, 150, 100)
    data_path = os.path.join(data_dir, 'option_data.csv')
    sample_data = generate_sample_data(S0, E, rf, sigma, T, S_range, data_path)
    
    # Plot option values
    option_values_path = os.path.join(figs_dir, 'option_values.png')
    plot_option_values(S_range, E, rf, sigma, T, save_path=option_values_path)
    
    # Plot Greeks
    plot_greeks(S_range, E, rf, sigma, T, save_dir=figs_dir)
    
    # Perform sensitivity analysis
    param_ranges = {
        'volatility': np.linspace(0.05, 0.5, 50),
        'time_to_expiry': np.linspace(0.1, 2, 50)
    }
    sensitivity_analysis(S0, E, rf, sigma, T, param_ranges, figs_dir)
    
    # Plot implied volatility surface
    strike_range = np.linspace(70, 130, 20)
    expiry_range = np.linspace(0.1, 2, 20)
    iv_surface_path = os.path.join(figs_dir, 'iv_surface.png')
    plot_iv_surface(strike_range, expiry_range, S0, rf, save_path=iv_surface_path)
    
    print("\nAnalysis complete!")
    print(f"Data saved to: {data_dir}")
    print(f"Figures saved to: {figs_dir}")
    
    # Display summary of calculations
    print("\nSummary of Black-Scholes Option Pricing:")
    print("=" * 50)
    print(f"Underlying Price: ${S0:.2f}")
    print(f"Strike Price: ${E:.2f}")
    print(f"Risk-free Rate: {rf:.2%}")
    print(f"Volatility: {sigma:.2%}")
    print(f"Time to Expiration: {T:.2f} years")
    print("-" * 50)
    print(f"Call Option Price: ${call_price:.4f}")
    print(f"Put Option Price: ${put_price:.4f}")
    print("-" * 50)
    print("Option Greeks:")
    print(f"Call Delta: {greeks['call_delta']:.4f}")
    print(f"Put Delta: {greeks['put_delta']:.4f}")
    print(f"Gamma: {greeks['gamma']:.6f}")
    print(f"Call Theta: ${greeks['call_theta']:.6f} per day")
    print(f"Put Theta: ${greeks['put_theta']:.6f} per day")
    print(f"Vega: ${greeks['vega']:.6f} per 1% change in volatility")
    print(f"Call Rho: ${greeks['call_rho']:.6f} per 1% change in interest rate")
    print(f"Put Rho: ${greeks['put_rho']:.6f} per 1% change in interest rate")
    print("=" * 50)
