import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from typing import Callable


def plot(*lines) -> None:
    """
    """
    [plt.plot(x, y, label=label, c=c) for (x, y, label), c in zip(lines, mcolors.BASE_COLORS)]
    plt.legend()
    plt.show()
    plt.close()
    return None


def scatter(*data) -> None:
    """
    """
    [plt.scatter(x, y, label=label, c=c) for (x, y, label), c in zip(data, mcolors.BASE_COLORS)]
    plt.legend()
    plt.show()
    plt.close()


def decision_boundary(X: np.ndarray[float], pred: Callable[[np.ndarray[float]], np.ndarray[int]]):
    """
    """
    d = X.shape[1]

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    n_points = xx.ravel().shape[0]

    grid_points = np.zeros((n_points, d))

    grid_points[:, 0] = xx.ravel()
    grid_points[:, 1] = yy.ravel()
    
    grid_points = grid_points.reshape(1, -1, d)

    Z = pred(grid_points[0])
    Z = Z.reshape(xx.shape)

    return xx, yy, Z


def boundary_comparison(X: np.ndarray, Y: np.ndarray, pred: Callable[[np.ndarray[float]], np.ndarray[int]], INIT: tuple[np.ndarray[float]], file: str = "", save: bool = False, show: bool = True) -> None:
    """
    """

    _, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].contourf(*INIT, alpha=0.3, cmap='bwr')
    ax[0].scatter(X[:, 0], X[:, 1], c=Y.flatten(), cmap='bwr', edgecolors='k')
    ax[0].set_xlabel("Feature 1")
    ax[0].set_ylabel("Feature 2")
    ax[0].set_title("Initialized Boundaries")

    ax[1].contourf(*decision_boundary(X, pred), alpha=0.3, cmap='bwr')
    ax[1].scatter(X[:, 0], X[:, 1], c=Y.flatten(), cmap='bwr', edgecolors='k')
    ax[1].set_xlabel("Feature 1")
    ax[1].set_ylabel("Feature 2")
    ax[1].set_title("Trained Boundaries")

    plt.tight_layout()

    if show:
        plt.show()

    if save:
        plt.savefig(file)

    plt.close()

    return
