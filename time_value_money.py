from math import exp

# Discrete
def future_dicrete_value(x, r, n):
    return x*(1+r)**n
def present_dicrete_value(x, r, n):
    return x*(1+r)**-n

# Continuous
def future_continous(x, r, t):
    return x*exp(r*t)

def present_continous(x, r, t):
    return x*exp(-r*t)

if __name__ == '__main__':
    # value of investment in dollars
    x = 100
    # define the interest rate
    r = 0.05
    # duration 
    n = 5 # yrs

    print("Future values of x (discrete): %s" %future_dicrete_value(x, r, n))
    print("Present values of x (discrete): %s" %present_dicrete_value(x, r, n))
    print("Future values of x (continuous): %s" %future_continous(x, r, n))
    print("Present values of x (continuous): %s" %present_continous(x, r, n))