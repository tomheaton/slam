import numpy as np
import matplotlib.pyplot as plt

def plot_graph(vertices, edges, save=False):
    # print("Plotting graph...")

    ############
    # Vertices #
    ############
    xs = []; ys = []
    for v in range(len(vertices)/3):
        x, y, _ = vertices[v*3:v*3+3]
        xs.append(x); ys.append(y)

    plt.scatter(xs, ys, s=10, marker='o', color='orange', zorder=1)

    #########
    # Edges #
    #########
    for edge in edges:
        i = edge.idFrom*3
        j = edge.idTo*3

        xy = np.array([vertices[i:i+3], vertices[j:j+3]])
        plt.plot(xy[:, 0], xy[:, 1], color='green', zorder=0)

    plt.title("GraphSLAM")
    plt.axis("off")

    if save:
        print("Saving graph...")
        plt.savefig("graph.pdf")
        print("Graph saved.")

    print("Showing graph...")
    plt.show()


