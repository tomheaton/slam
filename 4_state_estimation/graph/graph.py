from landmark import Landmark
from point import Point
from edge import Edge


class Graph():
    #landmarks: list[Landmark]
    #points: list[Point]
    #edges: list[Edge]

    def __init__(self, landmarks: list[Landmark] = None, points: list[Point] = None, edges: list[Edge] = None):
        if landmarks == None:
            self.landmarks = []
        else:
            self.landmarks = landmarks

        if points == None:
            self.points = []
        else:
            self.points = points

        if edges == None:
            self.edges = []
        else:
            self.edges = edges

    def get_connected_node_ids(self, node_id):
        connected_ids = []

        for edge in self.edges:
            other = edge.get_connected(node_id)

            if other != None:
                connected_ids.append(other)

        return connected_ids

    def get_node_from_id(self, node_id):
        for landmark in self.landmarks:
            if landmark.identifier == node_id:
                return landmark

        for point in self.points:
            if point.identifier == node_id:
                return point

        return None

    def is_landmark(self, node_id):
        for landmark in self.landmarks:
            if landmark.identifier == node_id:
                return True

        return False
