from uuid_generator import UUIDGenerator
import numpy as np

from point import Point


class Landmark(Point):
    #identifier: int
    #x: float
    #y: float

    def __init__(self, x: float, y: float, identifier: int = -1):
        super().__init__(x, y, identifier)

    def as_vector(self) -> np.ndarray:
        return np.array([[self.x], [self.y]])

    @staticmethod
    def from_vector(vector: np.ndarray):
        return Landmark(vector[0, 0], vector[1, 0])
