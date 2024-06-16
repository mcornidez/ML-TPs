import numpy as np
# from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from enum import Enum


class Method(Enum):
    MAX = 1
    MIN = 2
    CENTROID = 3
    AVERAGE = 4


class Cluster:
    def __init__(self, id, points, method):
        self.points = points
        self.method = method
        self.id = id
        self.distances = {}

    def join(self, other, id):
        points = np.concatenate((self.points, other.points), axis=0)
        assert self.method == other.method
        return Cluster(id, points, self.method)

    def distance(self, other):
        if other.id in self.distances:
            return self.distances[other.id]

        match self.method:
            case Method.MAX:
                d = self.distance_max(other)
            case Method.MIN:
                d = self.distance_min(other)
            case Method.AVERAGE:
                d = self.distance_avg(other)
            case Method.CENTROID:
                d = self.distance_centroid(other)
            case _:
                raise Exception("Non valid method")

        self.distances[other.id] = d
        other.distances[self.id] = d
        return d

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
        print("a")

        diff = get_diffs(self.points, other.points)

        return np.min(diff)

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


def train_hierarchical(data: np.ndarray, method: Method):
    linkage = []
    idx = len(data)
    clusters = {i: Cluster(i, np.array([row]), method) for i, row in enumerate(data)}

    print("c")
    for id in range(idx, 2 * idx - 1):
        print(id)
        values = np.array(list(clusters.values()))
        matrix = calculate_matrix(values)
        (i, j) = np.unravel_index(np.argmin(matrix), matrix.shape)
        d = values[i].distance(values[j])
        id_i = values[i].id
        id_j = values[j].id
        new = values[i].join(values[j], id)

        del clusters[id_i]
        del clusters[id_j]

        clusters[id] = new

        linkage.append([id_i, id_j, d, new.size()])

    return linkage


def calculate_matrix(clusters: np.ndarray):
    size = len(clusters)
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i, size):
            print("b")
            d = clusters[i].distance(clusters[j])

            if d == 0:
                d = np.inf

            matrix[i, j] = d
            matrix[j, i] = d

    return matrix

def get_diffs(p1, p2):

    p1 = p1.reshape((1, p1.shape[1], p1.shape[0]))
    p2 = p2.reshape((p1.shape[0], p1.shape[1], 1))

    return np.linalg.norm(p1 - p2, axis = 1)




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
# X = np.array(
#     [
#         [0.40, 0.53],
#         [0.22, 0.38],
#         [0.35, 0.32],
#         [0.26, 0.19],
#         [0.08, 0.41],
#         [0.45, 0.30],
#     ]
# )

# linkage2 = train_hierarchical(X, Method.MIN)


# print(np.array(linkage2))
# print(linkage(X, "single"))

# fig.update_layout(width=800, height=800)
