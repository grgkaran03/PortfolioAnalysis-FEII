import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import random

def plot_curve(x, y, color, label, linestyle='-'):
    plt.plot(x, y, color=color, label=label, linestyle=linestyle)

def plot_scatter(x, y, color, label, linestyle='-'):
    plt.scatter(x, y, color=color, label=label, linestyle=linestyle)

def calculate_weights(M, C, mu):
    n = len(M)
    C_inverse = np.linalg.inv(C)
    u = np.ones(n)
    p = [[1, u @ C_inverse @ np.transpose(M)], [mu, M @ C_inverse @ np.transpose(M)]]
    q = [[u @ C_inverse @ np.transpose(u), 1], [M @ C_inverse @ np.transpose(u), mu]]
    r = [[u @ C_inverse @ np.transpose(u), u @ C_inverse @ np.transpose(M)], [M @ C_inverse @ np.transpose(u), M @ C_inverse @ np.transpose(M)]]
    det_p, det_q, det_r = np.linalg.det(p), np.linalg.det(q), np.linalg.det(r)
    det_p /= det_r
    det_q /= det_r
    w = det_p * (u @ C_inverse) + det_q * (M @ C_inverse)
    return w

def minimum_variance_portfolio(M, C):
    u = np.ones(len(M))
    weight_min_var = u @ np.linalg.inv(C) / (u @ np.linalg.inv(C) @ np.transpose(u))
    mu_min_var = weight_min_var @ np.transpose(M)
    risk_min_var = math.sqrt(weight_min_var @ C @ np.transpose(weight_min_var))
    return risk_min_var, mu_min_var

def get_equation(x, y):
    slope, intercept = [], []
    for i in range(len(x) - 1):
        x1, x2 = x[i], x[i + 1]
        y1, y2 = y[i], y[i + 1]
        slope.append((y2 - y1) / (x2 - x1))
        intercept.append(y1 - slope[-1] * x1)
    return sum(slope) / len(slope), sum(intercept) / len(intercept)

def main():
    M = [0.1, 0.2, 0.15]
    C = [[0.005, -0.010, 0.004], [-0.010, 0.040, -0.002], [0.004, -0.002, 0.023]]

    returns = np.linspace(0, 0.5, num=10000)
    risk, actual_returns, weights = [], [], []
    risk_feasible_region, returns_feasible_region = [], []

    for mu in returns:
        w = calculate_weights(M, C, mu)
        if any(i < 0 for i in w):
            continue
        weights.append(w)
        sigma = math.sqrt(w @ C @ np.transpose(w))
        risk.append(sigma)
        actual_returns.append(mu)

    for _ in range(500):
        w = random.randint(1, 100, size=3).astype(float)
        w /= np.sum(w)

        returns_feasible_region.append(M @ np.transpose(w))
        risk_feasible_region.append(math.sqrt(w @ C @ np.transpose(w)))

    risk_min_var, mu_min_var = minimum_variance_portfolio(M, C)
    returns_plot1, risk_plot1, returns_plot2, risk_plot2 = [], [], [], []

    for i in range(len(actual_returns)):
        if actual_returns[i] >= mu_min_var:
            returns_plot1.append(actual_returns[i])
            risk_plot1.append(risk[i])
        else:
            returns_plot2.append(actual_returns[i])
            risk_plot2.append(risk[i])

    plot_curve(risk_plot1, returns_plot1, color='green', label='Efficient frontier', linestyle='-')
    plot_scatter(risk_feasible_region, returns_feasible_region, color='m', label='Feasible region', linestyle='-.')
    plot_scatter([risk_min_var], [mu_min_var], color='orange', label='Minimum Variance Point', linestyle=':')

    plt.xlabel("Risk (sigma)")
    plt.ylabel("Returns")
    plt.title("Minimum Variance Curve")

    plt.legend()
    plt.show()

    M_1 = [0.1, 0.2]
    C_1 = [[0.005, -0.010], [-0.010, 0.040]]
    risk_1, actual_returns_1, weights_1 = [], [], []

    for mu in returns:
        w = calculate_weights(M_1, C_1, mu)
        if any(i < 0 for i in w):
            continue
        weights_1.append(w)
        sigma = math.sqrt(w @ C_1 @ np.transpose(w))
        risk_1.append(sigma)
        actual_returns_1.append(mu)

    M_2 = [0.2, 0.15]
    C_2 = [[0.040, -0.002], [-0.002, 0.023]]
    risk_2, actual_returns_2, weights_2 = [], [], []

    for mu in returns:
        w = calculate_weights(M_2, C_2, mu)
        if any(i < 0 for i in w):
            continue
        weights_2.append(w)
        sigma = math.sqrt(w @ C_2 @ np.transpose(w))
        risk_2.append(sigma)
        actual_returns_2.append(mu)

    M_3 = [0.1, 0.15]
    C_3 = [[0.005, 0.004], [0.004, 0.023]]
    risk_3, actual_returns_3, weights_3 = [], [], []

    for mu in returns:
        w = calculate_weights(M_3, C_3, mu)
        if any(i < 0 for i in w):
            continue
        weights_3.append(w)
        sigma = math.sqrt(w @ C_3 @ np.transpose(w))
        risk_3.append(sigma)
        actual_returns_3.append(mu)

    plot_curve(risk, actual_returns, color='green', label='3 stocks', linestyle='-')
    plot_curve(risk_1, actual_returns_1, color='blue', label='Stock 1 and 2', linestyle='--')
    plot_curve(risk_2, actual_returns_2, color='orange', label='Stock 2 and 3', linestyle='-.')
    plot_curve(risk_3, actual_returns_3, color='red', label='Stock 1 and 3', linestyle=':')

    plt.xlabel("Risk (sigma)")
    plt.ylabel("Returns")
    plt.title("Minimum Variance Curve - No short sales")

    plt.legend()
    plt.show()

    plot_scatter(risk, actual_returns, color='green', label='3 stocks', linestyle='-')
    plot_scatter(risk_1, actual_returns_1, color='blue', label='Stock 1 and 2', linestyle='--')
    plot_scatter(risk_2, actual_returns_2, color='orange', label='Stock 2 and 3', linestyle='-.')
    plot_scatter(risk_3, actual_returns_3, color='red', label='Stock 1 and 3', linestyle=':')
    plot_scatter(risk_feasible_region, returns_feasible_region, color='m', label='Feasible region', linestyle='--')

    plt.xlabel("Risk (sigma)")
    plt.ylabel("Returns")
    plt.title("Minimum Variance Curve (with feasible region) - No short sales")

    plt.legend()
    plt.show()

    weights.clear()
    risk.clear()

    for mu in returns:
        w = calculate_weights(M, C, mu)
        weights.append(w)
        sigma = math.sqrt(w @ C @ np.transpose(w))
        risk.append(sigma)

    w_01, w_02, w_03 = np.array([i[0] for i in weights]), np.array([i[1] for i in weights]), np.array([i[2] for i in weights])
    x = np.linspace(-5, 5, 1000)
    y = np.zeros(len(x))

    m, c = get_equation(w_01, w_02)
    print("Eqn of line w1 vs w2 is:")
    print("w2 = {:.2f} w1 + {:.2f}".format(m, c))
    plt.axis([-0.5, 1.5, -0.5, 1.5])
    plot_curve(w_01, w_02, color='blue', label='w1 vs w2', linestyle='-')
    plot_curve(w_01, 1 - w_01, color='green', label='w1 + w2 = 1', linestyle='--')
    plot_curve(x, y, color='red', label='w2 = 0', linestyle='-.')
    plot_curve(y, x, color='yellow', label='w1 = 0', linestyle=':')

    plt.xlabel("weights_1 (w1)")
    plt.ylabel("weights_2 (w2)")
    plt.title("Weights corresponding to min variance curve (w1 vs w2)")

    plt.legend()
    plt.show()

    m, c = get_equation(w_02, w_03)
    print("Eqn of line w2 vs w3 is:")
    print("w3 = {:.2f} w2 + {:.2f}".format(m, c))
    plt.axis([-0.5, 1.5, -0.5, 1.5])
    plot_curve(w_02, w_03, color='blue', label='w2 vs w3', linestyle='-')
    plot_curve(w_02, 1 - w_02, color='green', label='w2 + w3 = 1', linestyle='--')
    plot_curve(x, y, color='red', label='w3 = 0', linestyle='-.')
    plot_curve(y, x, color='yellow', label='w2 = 0', linestyle=':')

    plt.xlabel("weights_2 (w2)")
    plt.ylabel("weights_3 (w3)")
    plt.title("Weights corresponding to min variance curve (w2 vs w3)")

    plt.legend()
    plt.show()

    m, c = get_equation(w_01, w_03)
    print("Eqn of line w1 vs w3 is:")
    print("w3 = {:.2f} w1 + {:.2f}".format(m, c))
    plt.axis([-0.5, 1.5, -0.5, 1.5])
    plot_curve(w_01, w_03, color='blue', label='w1 vs w3', linestyle='-')
    plot_curve(w_03, 1 - w_03, color='green', label='w1 + w3 = 1', linestyle='--')
    plot_curve(x, y, color='red', label='w3 = 0', linestyle='-.')
    plot_curve(y, x, color='yellow', label='w1 = 0', linestyle=':')

    plt.xlabel("weights_1 (w1)")
    plt.ylabel("weights_3 (w3)")
    plt.title("Weights corresponding to min variance curve (w1 vs w3)")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()