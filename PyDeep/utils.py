import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def plot(*lines) -> None:
    """
    """
    [plt.plot(x, label=label, c=c) for (x, label), c in zip(lines, mcolors.BASE_COLORS)]
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