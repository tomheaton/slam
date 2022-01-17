import numpy as np
import matplotlib.pyplot as plt

from graph import Graph


class GraphSlam:
    """Graph SLAM"""

    def __init__(self):
        self.graph = Graph()

    @staticmethod
    def f(x):
        return x ** 2 * np.exp(-x ** 2)

    x = np.linspace(start=0, stop=3, num=51)
    y = f(x)
    plt.plot(x, y)
    plt.show()
