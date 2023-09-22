import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from optimizers import *
from plane import *

# PLANE = "2D"
# PLANE = "Rosenbrock"
# PLANE = "SaddlePoint"
PLANE = "Sinusoidal"


ITER_NUM = 512
LEARNING_RATE = 0.5


if PLANE == "2D":
    grad_func = gradient_2d
    func = loss_function_2d
    x_init, y_init = 5.0, 5.0
    x_line, y_line = -6, 6

elif PLANE == "Rosenbrock":
    grad_func = rosenbrock_gradient
    func = rosenbrock_function
    x_init, y_init = -1.0, -1.0
    x_line, y_line = -3, 3

elif PLANE == "SaddlePoint":
    grad_func = saddle_point_gradient
    func = saddle_point_function
    x_init, y_init = 2.0, 2.0
    x_line, y_line = -6, 6

elif PLANE == "Sinusoidal":
    grad_func = sinusoidal_gradient
    func = sinusoidal_function
    x_init, y_init = 0.0, 0.0
    x_line, y_line = -6, 6

x_grid, y_grid = np.meshgrid(np.linspace(x_line, y_line, 100), np.linspace(x_line, y_line, 100))
z_grid = func(x_grid, y_grid)

## Initialize dictionaries to store paths for each optimizer
paths = {}
optimizer_functions = [gradient_descent, sgd_momentum, nesterov_momentum, adagrad, adadelta, adam]
for opt_func in optimizer_functions:
    if opt_func.__name__ == "adadelta":
        paths[opt_func.__name__] = opt_func(x_init, y_init, ITER_NUM, grad_func)
    else:
        paths[opt_func.__name__] = opt_func(x_init, y_init, ITER_NUM, LEARNING_RATE, grad_func)

## Initialize  3D plot
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


## create & nshow animation
anim = FuncAnimation(fig, update, frames=ITER_NUM + 1, init_func=init, blit=True)
plt.show()

## save animation
# anim.save('optimizer_paths_animation.mp4', writer='ffmpeg')
