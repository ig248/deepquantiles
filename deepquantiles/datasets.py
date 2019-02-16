import numpy as np


def bishop_s(num=1000, sort=True):
    y = np.linspace(0, 1, num=num)
    x = y + 0.3 * np.sin(2 * np.pi * y) + 0.2 * np.random.rand(num) - 0.1
    if sort:
        order = x.argsort()
        x, y = x[order], y[order]
    return x.reshape(-1, 1), y
