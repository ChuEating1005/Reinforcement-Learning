import numpy as np
from scipy.optimize import minimize
# Part A
a = np.array([90, -50, -40])
b = np.array([-9.8, 49, -39.2])
c = np.array([-9.5, -47.5, 57])
E = np.array([0.3, 0.5, -0.8])
Cov = 0.1 * np.outer(a - E, a - E) + 0.5 * np.outer(b - E, b - E) + 0.4 * np.outer(c - E, c - E)
print(f"Original Monte Carlo, Covariance Matrix:\n{Cov}")

# Part B
a_ = np.array([2.7, -1.5, -1.2])
b_ = np.array([-0.1, 0.5, -0.4])
c_ = np.array([0.2, 1, -1.2])
Cov_ = 0.1 * np.outer(a_ - E, a_ - E) + 0.5 * np.outer(b_ - E, b_ - E) + 0.4 * np.outer(c_ - E, c_ - E)
print(f"\nMonte Carlo with baseline, Covariance Matrix:\n{Cov_}")

# Part C
def trace_of_cov(base):
    a = (100 - base) * np.array([0.9, -0.5, -0.4])
    b = (98 - base) * np.array([-0.1, 0.5, -0.4])
    c = (95 - base) * np.array([-0.1, -0.5, 0.6])
    return np.trace(0.1 * np.outer(a - E, a - E) + 0.5 * np.outer(b - E, b - E) + 0.4 * np.outer(c - E, c - E))
base = minimize(trace_of_cov, x0=97)
print(f"\nOptimal base B(s) = {round(base.x[0],3)}")