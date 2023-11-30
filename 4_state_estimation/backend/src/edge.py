class Edge(object):

    def __init__(self, idFrom, idTo, estimate, info):
        self.idFrom = idFrom
        self.idTo = idTo
        self.estimate = estimate
        self.info = info
    
    def __repr__(self):
        return ("Edge: {}, {}, {}, {}").format(self.idFrom, self.idTo, self.estimate, self.info)

    def __str__(self):
        return ("Edge: {}, {}, {}, {}").format(self.idFrom, self.idTo, self.estimate, self.info)

