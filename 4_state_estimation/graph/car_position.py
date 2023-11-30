from uuid_generator import UUIDGenerator
from point import Point


class CarPosition(Point):
    #theta: float

    def __init__(self, x: float, y: float, theta: float, identifier: int = None):
        super().__init__(x, y, identifier)
        self.theta = theta
