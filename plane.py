import numpy as np


def loss_function_2d(x, y):
    return x**2 + y**2


def gradient_2d(x, y):
    return np.array([2 * x, 2 * y])


def rosenbrock_function(x, y, a=1, b=100):
    return (a - x) ** 2 + b * (y - x**2) ** 2


def rosenbrock_gradient(x, y, a=1, b=100):
    return np.array([-2 * (a - x) - 4 * b * x * (y - x**2), 2 * b * (y - x**2)])


def saddle_point_function(x, y):
    return x**2 - y**2


def saddle_point_gradient(x, y):
    return np.array([2 * x, -2 * y])


def sinusoidal_function(x, y):
    return np.sin(0.5 * x) * np.cos(0.5 * y)


def sinusoidal_gradient(x, y):
    grad_x = 0.5 * np.cos(0.5 * x) * np.cos(0.5 * y)
    grad_y = -0.5 * np.sin(0.5 * x) * np.sin(0.5 * y)
    return np.array([grad_x, grad_y])
