import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tabulate import tabulate as tab
from scipy.interpolate import CubicSpline

def calculate_weights(m, c, mu):
    cinv = np.linalg.inv(c)
    u = np.ones(len(m))
    num1 = [[1, u @ cinv @ np.transpose(m)], [mu, m @ cinv @ np.transpose(m)]]
    num2 = [[u @ cinv @ np.transpose(u), 1], [m @ cinv @ np.transpose(u), mu]]
    deno = [[u @ cinv @ np.transpose(u), u @ cinv @ np.transpose(m)], [m @ cinv @ np.transpose(u), m @ cinv @ np.transpose(m)]]
    det_1, det_2, det_d = np.linalg.det(num1), np.linalg.det(num2), np.linalg.det(deno)
    det_1 /= det_d
    det_2 /= det_d
    w = det_1 * (u @ cinv) + det_2 * (m @ cinv)
    return w

def calculate_mvp(M, C):
    u = np.ones(len(M))
    w = u @ np.linalg.inv(C) / (u @ np.linalg.inv(C) @ np.transpose(u))
    mu = w @ np.transpose(M)
    risk = math.sqrt(w @ C @ np.transpose(w))
    return risk, mu

def generate_efficient_frontier(M, C, ret):
    risk = []
    for mu in ret:
        w = calculate_weights(M, C, mu)
        sigma = math.sqrt(w @ C @ np.transpose(w))
        risk.append(sigma)
    return risk

def plot_efficient_frontier(risk1, ret1, risk2, ret2, rmv, mumv):
    plt.figure(figsize=(12, 8))
    
    plt.scatter(risk1, ret1, color='skyblue', marker='o', label='Efficient Frontier (High Return)', s=15)
    plt.scatter(risk2, ret2, color='lightcoral', marker='x', label='Efficient Frontier (Low Return)', s=15)
    
    plt.xlabel("Risk (Standard Deviation)")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier and Minimum Variance Portfolio")
    
    plt.scatter(rmv, mumv, color='green', marker='s', label='Minimum Variance Portfolio', s=100)
    plt.annotate(f'MVP\n({round(rmv, 4)}, {round(mumv, 4)})',
                 xy=(rmv, mumv), xytext=(rmv + 0.03, mumv), arrowprops=dict(facecolor='black', arrowstyle='->'))
    
    plt.legend()
    plt.show()

def generate_random_portfolios(M, C, mumv):
    mu10 = [random.uniform(mumv, 0.5) for _ in range(10)]
    mu10.sort()
    w10 = np.array([calculate_weights(M, C, mu) for mu in mu10])
    s10 = [np.sqrt(np.dot(w10[i], np.dot(C, w10[i]))) for i in range(10)]

    rows = [[np.array([w10[i]]).T, mu10[i], s10[i]] for i in range(10)]
    return rows

def print_portfolios(rows):
    table = tab(rows, ['Weights', 'Return', 'Risk'], tablefmt='grid')
    print(table)

def main():
    M = [0.1, 0.2, 0.15]
    C = [[0.005, -0.010, 0.004], [-0.010, 0.040, -0.002], [0.004, -0.002, 0.023]]

    ret = np.linspace(0, 0.5, 10000)

    risk = generate_efficient_frontier(M, C, ret)

    rmv, mumv = calculate_mvp(M, C)

    ret1, risk1, ret2, risk2 = [], [], [], []

    for i in range(len(ret)):
        if ret[i] >= mumv:
            ret1.append(ret[i])
            risk1.append(risk[i])
        else:
            ret2.append(ret[i])
            risk2.append(risk[i])

    
    plot_efficient_frontier(risk1, ret1, risk2, ret2, rmv, mumv)

    print("\nFor Part B: \n")

    rows = generate_random_portfolios(M, C, mumv)
    print_portfolios(rows)

    print("\nFor Part C: ")

    spl = CubicSpline(risk1, ret1)
    print("\nFor 15% Risk: ")
    print("\nMaximum Return is : {:.2%}".format(spl(0.15)))
    print("And the corresponding weights are : ", calculate_weights(M, C, spl(0.15)))

    risk2.sort()
    ret2.sort(reverse=True)
    spl = CubicSpline(risk2, ret2)
    print("\nMinimum Return is : {:.2%}".format(spl(0.15)))
    print("And the corresponding weights are : ", calculate_weights(M, C, spl(0.15)))

    print("\nFor Part D: ")

    print("\nFor 18% return: ")
    w18 = calculate_weights(M, C, 0.18)
    sigma18 = math.sqrt(w18 @ C @ np.transpose(w18))
    print("\nMinimum Risk is: {:.4f}".format(sigma18))
    print("And the corresponding weights are : ", w18)

    print("\nFor Part E: ")

    mu_rf = 0.1
    u = np.array([1, 1, 1])

    wmar = (M - mu_rf * u) @ np.linalg.inv(C) / ((M - mu_rf * u) @ np.linalg.inv(C) @ np.transpose(u))
    mumar = wmar @ np.transpose(M)
    rmar = math.sqrt(wmar @ C @ np.transpose(wmar))

    print("\nMarket Portfolio Weights = ", wmar)
    print("Return = {:.2%}".format(mumar))
    print("Risk = {:.2%}".format(rmar))

    retcml = []
    rcml = np.linspace(0, 1, num=10000)
    for i in rcml:
        retcml.append(mu_rf + (mumar - mu_rf) * i / rmar)

    slope, intercept = (mumar - mu_rf) / rmar, mu_rf

    print("\nEquation of CML is:")
    print("y = {:.3f} x + {:.3f}\n".format(slope, intercept))

    plt.scatter(rmar, mumar, color='red', s=50, label='Market portfolio')
    plt.plot(risk, ret, color='blue', label='Minimum variance curve', linewidth=2)
    plt.plot(rcml, retcml, color='yellow', label='Capital Market Line', linewidth=2)

    plt.xlabel("Risk (sigma)")
    plt.ylabel("Returns")
    plt.title("Capital Market Line with Minimum variance curve")
    plt.legend()
    plt.show()

    print("\nFor Part F: ")

    def calculate_risk_and_weights(sigma):
        muc = (mumar - mu_rf) * sigma / rmar + mu_rf
        wrf = (muc - mumar) / (mu_rf - mumar)
        wrisk = (1 - wrf) * wmar
        return wrf, wrisk, muc

    sigma_values = [0.1, 0.25]

    for sigma in sigma_values:
        wrf, wrisk, muc = calculate_risk_and_weights(sigma)

        print("\nRisk = {:.2%}".format(sigma))
        print("Risk-free weights =", wrf)
        print("Risky Weights =", wrisk)
        print("Returns = {:.2%}".format(muc))


if __name__ == "__main__":
    main()
