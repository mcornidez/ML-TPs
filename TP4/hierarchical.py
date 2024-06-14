from matplotlib.pylab import axis
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import numpy as np
from enum import Enum


class Method(Enum):
    MAX = 1
    MIN = 2
    CENTROID = 3
    AVERAGE = 4


class Cluster:
    def __init__(self, points):
        self.points = points

    def size(self):
        return len(self.points)

    def distance_max(self, other):
        p1 = self.points
        p2 = other.points

        distance = 0
        for p in p1:
            d = max(np.linalg.norm(p - p2, axis=1))
            if d > distance:
                distance = d

        return distance

    def distance_min(self, other):
        p1 = self.points
        p2 = other.points

        distance = np.inf
        for p in p1:
            d = min(np.linalg.norm(p - p2, axis=1))
            if d < distance:
                distance = d

        return distance

    def distance_avg(self, other):
        p1 = self.points
        p2 = other.points

        avg = None
        for p in p1:
            d = np.mean(np.linalg.norm(p - p2, axis=1))
            if avg is not None:
                avg = (avg + d) / 2

        return avg

    def distance_centroid(self, other):
        p1 = self.points
        p2 = other.points

        return np.linalg.norm(np.mean(p1, axis=0) - np.mean(p2, axis=0))

# X = np.random.rand(5, 5)
# X = np.array(
#     [
#         [0, 0.23, 0.22, 0.37, 0.34, 0.24],
#         [0.23, 0, 0.14, 0.19, 0.14, 0.24],
#         [0.22, 0.14, 0, 0.16, 0.28, 0.10],
#         [0.37, 0.19, 0.16, 0, 0.28, 0.22],
#         [0.34, 0.14, 0.28, 0.28, 0, 0.39],
#         [0.24, 0.24, 0.1, 0.22, 0.39, 0],
#     ]
# )
X = np.array(
    [
        [0.40, 0.53],
        [0.22, 0.38],
        [0.35, 0.32],
        [0.26, 0.19],
        [0.08, 0.41],
        [0.45, 0.30],
    ]
)

def train_hierarchical(data: np.ndarray):
    linkage = []
    
    data = np.array([np.insert(row, 0, i) for i, row in enumerate(data)])


# fig = ff.create_dendrogram(X)
# fig.update_layout({"width": 800, "height": 800})
# fig.show()


# The text for the leaf nodes is going to be big so force
# a rotation of 90 degrees.
# dendrogram(linkage(X, 'single'), leaf_rotation=90)

# fig.update_layout(width=800, height=800)
