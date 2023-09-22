import numpy as np



def gradient_descent(x_init, y_init, num_iterations, learning_rate, grad_func, **kwargs):
    x, y = x_init, y_init
    path = [(x, y)]
    for _ in range(num_iterations):
        grad_x, grad_y = grad_func(x, y, **kwargs)
        x -= learning_rate * grad_x
        y -= learning_rate * grad_y
        path.append((x, y))
    return path


def sgd_momentum(x_init, y_init, num_iterations, learning_rate, grad_func, beta=0.9, **kwargs):
    x, y = x_init, y_init
    velocity = np.array([0.0, 0.0])
    path = [(x, y)]
    for _ in range(num_iterations):
        grad = grad_func(x, y, **kwargs)
        velocity = beta * velocity - learning_rate * grad
        x, y = np.array([x, y]) + velocity
        path.append((x, y))
    return path


def nesterov_momentum(x_init, y_init, num_iterations, learning_rate, grad_func, beta=0.9, **kwargs):
    x, y = x_init, y_init
    velocity = np.array([0.0, 0.0])
    path = [(x, y)]
    for _ in range(num_iterations):
        grad = grad_func(x + beta * velocity[0], y + beta * velocity[1], **kwargs)
        velocity = beta * velocity - learning_rate * grad
        x, y = np.array([x, y]) + velocity
        path.append((x, y))
    return path


def adagrad(x_init, y_init, num_iterations, learning_rate, grad_func, epsilon=1e-8, **kwargs):
    x, y = x_init, y_init
    grad_squared = np.array([0.0, 0.0])
    path = [(x, y)]
    for _ in range(num_iterations):
        grad = grad_func(x, y, **kwargs)
        grad_squared += grad**2
        update = -learning_rate * grad / (np.sqrt(grad_squared) + epsilon)
        x, y = np.array([x, y]) + update
        path.append((x, y))
    return path


def adadelta(x_init, y_init, num_iterations, grad_func, rho=0.9, epsilon=1e-8, **kwargs):
    x, y = x_init, y_init
    grad_squared = np.array([0.0, 0.0])
    delta_squared = np.array([0.0, 0.0])
    path = [(x, y)]
    for _ in range(num_iterations):
        grad = grad_func(x, y, **kwargs)
        grad_squared = rho * grad_squared + (1 - rho) * grad**2
        delta = -np.sqrt(delta_squared + epsilon) / np.sqrt(grad_squared + epsilon) * grad
        delta_squared = rho * delta_squared + (1 - rho) * delta**2
        x, y = np.array([x, y]) + delta
        path.append((x, y))
    return path


def adam(x_init, y_init, num_iterations, learning_rate, grad_func, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
    x, y = x_init, y_init
    m = np.array([0.0, 0.0])
    v = np.array([0.0, 0.0])
    path = [(x, y)]
    for i in range(1, num_iterations + 1):
        grad = grad_func(x, y, **kwargs)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**i)
        v_hat = v / (1 - beta2**i)
        update = -learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        x, y = np.array([x, y]) + update
        path.append((x, y))
    return path
