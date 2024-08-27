import numpy as np
import math
import matplotlib.pyplot as plt
import time


class LookbackOption:
    def __init__(self, S0, T, r, sigma):
        self.S0 = S0
        self.T = T
        self.r = r
        self.sigma = sigma

    def arbitrage_condition(self, u, d, t):
        return not (d < math.exp(self.r * t) and math.exp(self.r * t) < u)

    def cache(self, idx, u, d, p, R, M, stock_price, running_max, option_prices):
        if idx == M + 1 or (stock_price, running_max) in option_prices[idx]:
            return

        self.cache(idx + 1, u, d, p, R, M, stock_price * u, max(stock_price * u, running_max), option_prices)
        self.cache(idx + 1, u, d, p, R, M, stock_price * d, max(stock_price * d, running_max), option_prices)

        if idx == M:
            option_prices[M][(stock_price, running_max)] = max(running_max - stock_price, 0)
        else:
            option_prices[idx][(stock_price, running_max)] = (
                    p * option_prices[idx + 1][(u * stock_price, max(u * stock_price, running_max))] +
                    (1 - p) * option_prices[idx + 1][(d * stock_price, running_max)]) / R

    def lookback_option_efficient(self, M, display):
        if display == 1:
            print("\n\n*********  Executing for M = {}  *********\n".format(M))
        curr_time_1 = time.time()

        u, d = 0, 0
        t = self.T / M

        u = math.exp(self.sigma * math.sqrt(t) + (self.r - 0.5 * self.sigma * self.sigma) * t)
        d = math.exp(-self.sigma * math.sqrt(t) + (self.r - 0.5 * self.sigma * self.sigma) * t)

        R = math.exp(self.r * t)
        p = (R - d) / (u - d)
        result = self.arbitrage_condition(u, d, t)

        option_prices = []
        for i in range(0, M + 1):
            option_prices.append(dict())

        self.cache(0, u, d, p, R, M, self.S0, self.S0, option_prices)

        if result:
            if display == 1:
                print("Arbitrage Opportunity exists for M = {}".format(M))
            return 0, 0
        else:
            if display == 1:
                print("No arbitrage exists for M = {}".format(M))

        if display == 1:
            print("Initial Price of Lookback Option \t= {}".format(option_prices[0][(self.S0, self.S0)]))
            print("Execution Time \t\t\t\t= {} sec\n".format(time.time() - curr_time_1))

        if display == 2:
            for i in range(len(option_prices)):
                print("At t = {}".format(i))
                for key, value in option_prices[i].items():
                    print("Intermediate state = {}\t\tPrice = {}".format(key, value))
                print()

        return option_prices[0][(self.S0, self.S0)]


def plot_fixed(x, y, x_axis, y_axis, title):
    plt.plot(x, y)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.show()

print("-----------------------  sub-part(a)  -----------------------")
M_values = [5, 10, 25, 50]
prices_a = []

lookback_a = LookbackOption(S0=100, T=1, r=0.08, sigma=0.30)

for m in M_values:
    prices_a.append(lookback_a.lookback_option_efficient(M=m, display=1))

# sub-part (b)
print("\n\n-----------------------  sub-part(b)  -----------------------")
plot_fixed(M_values, prices_a, "M", "Initial Option Prices", "Initial Option Prices vs M")
M_values_b = [i for i in range(1, 21)]
prices_b = []

lookback_b = LookbackOption(S0=100, T=1, r=0.08, sigma=0.30)

for m in M_values_b:
    prices_b.append(lookback_b.lookback_option_efficient(M=m, display=0))

plot_fixed(M_values_b, prices_b, "M", "Initial Option Prices", "Initial Option Prices vs M (Variation with more data-points for M)")

# sub-part (c)
print("\n\n-----------------------  sub-part(c)  -----------------------")
lookback_c = LookbackOption(S0=100, T=1, r=0.08, sigma=0.30)
lookback_c.lookback_option_efficient(M=5, display=2)