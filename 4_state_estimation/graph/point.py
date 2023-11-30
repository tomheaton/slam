from uuid_generator import UUIDGenerator
from abc import ABC


class Point(ABC):
    #identifier: int
    #x: float
    #y: float
    #theta: float

    def __init__(self, x: float, y: float, identifier: int = None):
        if identifier == None:
            self.identifier = UUIDGenerator.generate_id()
        else:
            self.identifier = identifier

        self.x = x
        self.y = y
