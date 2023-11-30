from enum import Enum
import numpy as np
class ConeTypes(Enum):
    BLUE_CONES          = 0
    YELLOW_CONES        = 1
    ORANGE_CONES        = 2
    BIG_ORANGE_CONES    = 3
    UNKNOWN_COLOR_CONES = 4

# class VertexTypes(Enum):
    # POSE                = 0
    # BLUE_CONES          = 1
    # YELLOW_CONES        = 2
    # ORANGE_CONES        = 3
    # BIG_ORANGE_CONES    = 4
    # UNKNOWN_COLOR_CONES = 5

# class Vertex(np.ndarray):
    # # This is how the numpy docs tell you to subclass a np.ndarray
    # def __new__(cls, x: float, y: float, yaw: float, vertexType: VertexTypes):
        # obj = np.array([x,y,yaw], dtype=np.float32).view(cls)
        # obj.vertexType: VertexTypes = vertexType
        # return obj

    # def to_matrix(self):
        # # If it's a pose return a rotation+translation matrix
        # if self.vertexType == VertexTypes.POSE:
            # return np.array([
                # [np.cos(self[2]), -np.sin(self[2]), self[0]],
                # [np.sin(self[2]),  np.cos(self[2]), self[1]],
                # [0, 0, 1]
            # ])
        # # If it's a landmark(cone) return matrix with no rotation
        # return np.array([
            # [0, 0, self[0]],
            # [0, 0, self[1]],
            # [0, 0, 1]
        # ])

def v2t(v):
    """(x,y) to transfomration (3x3) matrix"""
    return np.array([
        [1, 0, v[0]],
        [0, 1, v[1]],
        [0, 0, 1]
    ])

def se2v2t(v):
    """SE(2) (x,y,yaw) to transformation (3x3) matrix"""
    return np.array([
        [np.cos(v[2]), -np.sin(v[2]), v[0]],
        [np.sin(v[2]),  np.cos(v[2]), v[1]],
        [0, 0, 1]
    ])
