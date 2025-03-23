class ZeroCouponBonds:
    
    def __init__(self, principal, maturity, interest_rate):
        self.principal = principal # principal amount
        self.maturity = maturity # date to maturity
        self.interest_rate = interest_rate/100 # market interest rate (discounting)

    def present_value(self, x, n):
        return  x / (1 + self.interest_rate)**n

    def calculate_price(self):
        return self.present_value(self.principal, self.maturity)

if __name__ == '__main__':
    bond = ZeroCouponBonds(1000, 2, 4)
    print("Price of the bond: $%.2f" %round(bond.calculate_price(), 3))