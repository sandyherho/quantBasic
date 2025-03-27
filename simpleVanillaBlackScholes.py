import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use("bmh")

def call_option_price(S, E, rf, sigma, T, t=0):
    # 1st calculate d1 and d2 parameters
    d1 = (np.log(S/E) + (rf + (sigma**2)/2)*(T-t)) / (sigma * np.sqrt(T-t))
    d2 = d1 - sigma * np.sqrt(T - t)
    print("The d1 & d2 parameters: (%s, %s)" % (d1, d2))
    # use the N(x) to calculate the price of the option
    return S*stats.norm.cdf(d1) - E*np.exp(-rf*(T-t))*stats.norm.cdf(d2)

def put_option_price(S, E, rf, sigma, T, t=0):
    # 1st calculate d1 and d2 parameters
    d1 = (np.log(S/E) + (rf + (sigma**2)/2)*(T-t)) / (sigma * np.sqrt(T-t))
    d2 = d1 - sigma * np.sqrt(T - t)
    print("The d1 & d2 parameters: (%s, %s)" % (d1, d2))
    # use the N(x) to calculate the price of the option
    return E*np.exp(-rf*(T-t))*stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1)

if __name__=='__main__':
    # underlying stock price at t=0
    S0 = 100
    # strike price
    E = 100
    # Expiry 1 yr = 365 days
    T = 1
    # risk-free rate 
    rf = 0.05
    # volatility of the underlying stocks
    sigma = 0.2
    
    print("Call option price according to Black Scholes model: ", call_option_price(S0, E, rf, sigma, T))
    print("Put option price according to Black Scholes model: ", put_option_price(S0, E, rf, sigma, T))