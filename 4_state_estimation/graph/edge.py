import numpy as np


class Edge():
    id1: int
    id2: int

    # Tuple (x, y, theta)
    estimate: tuple[float, float, float]

    # numpy 3x3 matrix
    information_matrix: np.ndarray

    def __init__(self, id1: int, id2: int, estimate: tuple[float, float, float] = None, information_matrix: np.ndarray = None):
        self.id1 = id1
        self.id2 = id2

        self.estimate = estimate
        self.information_matrix = information_matrix

    def get_connected(self, node_id):
        if self.id1 == node_id:
            return self.id2
        elif self.id2 == node_id:
            return self.id1
        else:
            return None
