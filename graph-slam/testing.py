import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    def f(x):
        return x ** 2 * np.exp(-x ** 2)

    x = np.linspace(start=0, stop=3, num=51)
    y = f(x)

    # plt.plot(x, y)
    plt.plot(4, 5, "bo")
    plt.xlabel('x meters')
    plt.ylabel('y meters')
    plt.title("Graph SLAM")
    plt.show()
