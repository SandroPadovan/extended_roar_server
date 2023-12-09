import math
import os

from matplotlib import use as use_plot_backend
import matplotlib.pyplot as plt
import numpy as np

from environment.state_handling import initialize_storage, cleanup_storage, get_storage_path


def pos(r, h, s_1, s_2):
    # hidden
    return list(map(lambda v: s_1 * math.log(s_2 * v + 1) + abs(h), r))


def neg(r, d):
    # detected
    return list(map(lambda v: (d/max(v, 1)) - abs(d), r))


try:
    initialize_storage()
    use_plot_backend("template")

    x_lim = 500
    description = "variant1+5"
    x = np.linspace(0, x_lim)

    plt.plot(x, pos(x, 0, 10, 1), color="tab:blue", linestyle="dashed")     # variant 1 hidden
    plt.plot(x, pos(x, 0, 100, 0.01), color="tab:blue")     # variant 5 hidden
    plt.plot(x, neg(x, -20), color="tab:orange")        # variant 1 / 5 detected

    plt.ylabel("Reward")
    plt.xlabel("Encryption Rate")
    plt.legend(["Hidden Variant 1", "Hidden Variant 5", "Detected Variant 1 / 5"])

    fig_file = os.path.join(get_storage_path(), "reward-func={}.png".format(description))
    plt.savefig(fig_file)
finally:
    cleanup_storage()
