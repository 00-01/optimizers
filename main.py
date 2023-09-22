import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from optimizers import *
from plane import *

# "2D" or "Rosenbrock"
# PLANE = "2D"
PLANE = "Rosenbrock"


def loss_function_2d(x, y):
    return x**2 + y**2


def rosenbrock_function(x, y, a=1, b=100):
    return (a - x) ** 2 + b * (y - x**2) ** 2  # Initialize parameters


def gradient_2d(x, y):
    return np.array([2 * x, 2 * y])


def rosenbrock_gradient(x, y, a=1, b=100):
    return np.array([-2 * (a - x) - 4 * b * x * (y - x**2), 2 * b * (y - x**2)])


if PLANE == "2D":
    grad_func = gradient_2d
    func = loss_function_2d

    x_init, y_init = 5.0, 5.0
    num_iterations = 50  # Number of iterations
    learning_rate = 0.1  # Learning rate for most optimizers

    x_line, y_line = -6, 6
elif PLANE == "Rosenbrock":
    grad_func = rosenbrock_gradient
    func = rosenbrock_function

    x_init, y_init = -1.0, -1.0
    num_iterations = 100
    learning_rate = 0.005

    x_line, y_line = -3, 3
x_grid, y_grid = np.meshgrid(np.linspace(x_line, y_line, 100), np.linspace(x_line, y_line, 100))
z_grid = func(x_grid, y_grid)

## Initialize dictionaries to store paths for each optimizer
paths = {}
optimizer_functions = [gradient_descent, sgd_momentum, nesterov_momentum, adagrad, adadelta, adam]
for opt_func in optimizer_functions:
    if opt_func.__name__ == "adadelta":
        paths[opt_func.__name__] = opt_func(x_init, y_init, num_iterations, grad_func)
    else:
        paths[opt_func.__name__] = opt_func(x_init, y_init, num_iterations, learning_rate, grad_func)

## Initialize the 3D plot for animation
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 10))

## Initialize lines for each optimizer
lines = {}
for name in paths.keys():
    (line,) = ax.plot([], [], [], marker="o", label=name)
    lines[name] = line


## Initialization function for animation
def init():
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, cmap="viridis")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Loss")
    ax.set_title(f"Optimizer Paths on {PLANE} Surface")
    ax.legend()
    if PLANE == "Rosenbrock":
        ax.set_zlim([0, 2000])
    return lines.values()


## Update function for animation
def update(frame):
    for name, line in lines.items():
        path = paths[name]
        x_vals, y_vals = zip(*path[: frame + 1])
        z_vals = [func(x, y) for x, y in path[: frame + 1]]
        line.set_data(x_vals, y_vals)
        line.set_3d_properties(z_vals)
    return lines.values()


## Create the animation
anim = FuncAnimation(fig, update, frames=num_iterations + 1, init_func=init, blit=True)
plt.show()

## save animation
# anim.save('optimizer_paths_animation.mp4', writer='ffmpeg')
