import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import use as use_plot_backend

from environment.state_handling import initialize_storage, cleanup_storage, get_storage_path


def relu(x):
    return list(map(lambda v: v if v > 0 else 0, x))


def logistic(x):
    return 1 / (1 + np.exp(-x))


def silu(x):
    return x / (1 + np.exp(-x))


try:
    initialize_storage()
    use_plot_backend("template")  # required for matplotlib, throws error otherwise

    x_lim = 4
    description = "a1-{}--a2-{}--x-{}".format("Logistic", "SiLU", x_lim)

    x = np.linspace(-x_lim, x_lim, num=500)

    # plt.plot(x, relu(x))
    plt.plot(x, logistic(x))
    plt.plot(x, silu(x))

    plt.legend(["Logistic", "SiLU"])

    plt.xlim(-x_lim, x_lim)
    plt.grid()

    fig_file = os.path.join(get_storage_path(), "activation-func={}.png".format(description))
    plt.savefig(fig_file)
finally:
    cleanup_storage()
