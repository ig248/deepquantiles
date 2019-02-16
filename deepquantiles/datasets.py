import numpy as np


def bishop_s(num=1000, sort=True):
    y = np.linspace(0, 1, num=num)
    x = y + 0.3 * np.sin(2 * np.pi * y) + 0.2 * np.random.rand(num) - 0.1
    if sort:
        order = x.argsort()
        x, y = x[order], y[order]
    return x.reshape(-1, 1), y


def xmas_tree(num=1000, n_branches=4, sort=True):
    y = np.linspace(0, 1, num=num)
    x = (np.random.rand(num) - 0.5) * (0.1 + y % (1 / n_branches)) * y * n_branches
    if sort:
        order = x.argsort()
        x, y = x[order], y[order]
    return x.reshape(-1, 1), y


def skewed_heteroscedastic(num=1000, with_mean=True, sort=True):
    x = np.linspace(0, 1, num=num)
    y = np.random.exponential(scale=x, size=x.size)
    if with_mean:
        mean = 4 * x * (1 - x)
        y = mean + y
    return x.reshape(-1, 1), y


def skewed_heteroscedastic_mean(X):
    """Theoretical mean."""
    mean = 4 * X * (1 - X) + X
    return mean


def skewed_heteroscedastic_quantiles(X, q):
    """Theoretical quantiles."""
    y_q = -np.log(1 - q) * X + 4 * X * (1 - X)
    return y_q
