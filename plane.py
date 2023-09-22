import numpy as np



    
def loss_function_2d(x, y):
    return x**2 + y**2


def gradient_2d(x, y):
    return np.array([2 * x, 2 * y])


def rosenbrock_function(x, y, a=1, b=100):
    return (a - x) ** 2 + b * (y - x**2) ** 2


def rosenbrock_gradient(x, y, a=1, b=100):
    return np.array([-2 * (a - x) - 4 * b * x * (y - x**2), 2 * b * (y - x**2)])
