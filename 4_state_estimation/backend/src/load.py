import numpy as np
from edge import Edge

# Debug
import math

def load_graph_file(file):
    """Loads g2o/toro file"""

    vertices = []
    edges = []

    # Reference: https://www.dropbox.com/s/uwwt3ni7uzdv1j7/g2oVStoro.pdf?dl=0

    with open(file, 'r') as f:
        for line in f.readlines():
            items = line.split()
            if items[0] == "VERTEX_SE2":
                _, ID, x, y, theta = items

                vertices.append(np.float64(x))
                vertices.append(np.float64(y))
                vertices.append(np.float64(theta))

                # Debug
                # x = np.float64(x)
                # y = np.float64(y)
                # theta = np.float64(theta)

                # if x > 0 and x < 0.001:
                    # x = 0
                # if x < 0 and x > -0.001:
                    # x = 0

                # if y > 0 and y < 0.001:
                    # y = 0
                # if y < 0 and y > -0.001:
                    # y = 0

                # if theta > 0 and theta < 0.001:
                    # theta = 0
                # if theta < 0 and theta > -0.001:
                    # theta = 0

                # vertices.append(x)
                # vertices.append(y)
                # vertices.append(theta)

                # assert not math.isnan(float(theta))

                continue

            elif items[0] == "EDGE_SE2":
                _, idFrom, idTo, dx, dy, dth, I11, I12, I13, I22, I23, I33 = items

                idFrom = int(idFrom)
                idTo   = int(idTo)
                estimate = np.array([dx,dy,dth], dtype=np.float64)
                info = np.array([
                    [I11, I12, I13],
                    [I12, I22, I23],
                    [I13, I23, I33]
                ], dtype=np.float64)

                edges.append(Edge(idFrom, idTo, estimate, info))

                continue

    return np.array(vertices, dtype=np.float64), edges
