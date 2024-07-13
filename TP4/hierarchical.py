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
    def __init__(self, id, points, method, genres_count):
        self.points = points
        self.genres_count = genres_count
        self.method = method
        self.id = id
        self.distances = {}

    def join(self, other, id):
        points = np.concatenate((self.points, other.points), axis=0)
        assert self.method == other.method
        return Cluster(id, points, self.method, self.genres_count + other.genres_count)

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
        diff = get_diffs(self.points, other.points)
        return np.max(diff)

    def distance_min(self, other):
        diff = get_diffs(self.points, other.points)
        return np.min(diff)

    def distance_avg(self, other):
        diff = get_diffs(self.points, other.points)
        return np.mean(diff)

    def distance_centroid(self, other):
        return np.linalg.norm(np.mean(self.points, axis=0) - np.mean(other.points, axis=0))


def train_hierarchical(data: np.ndarray, method: Method, genres):
    linkage = []
    idx = len(data)
    unique = np.unique(genres)
    genres_count = np.zeros((len(genres), len(unique)))
    for i, j in enumerate(genres):
        genres_count[i, j] = 1
    clusters = {i: Cluster(i, np.array([row]), method, genres_count[i]) for i, row in enumerate(data)}

    for id in range(idx, 2 * idx - 1):
        print(id)
        values = np.array(list(clusters.values()))
        if 2 * idx - 1 - id < 30:
            for clust in values:
                print(clust.genres_count)
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
    diag = np.arange(size)
    dists = vect_get_dist(np.reshape(clusters, (size, 1)), clusters)
    dists[diag, diag] = np.inf
    return dists

def get_diffs(p1, p2):

    p1 = p1.reshape((1, p1.shape[1], p1.shape[0]))
    p2 = p2.reshape((p2.shape[0], p2.shape[1], 1))

    return np.linalg.norm(p1 - p2, axis = 1)

def get_dist(clust1, clust2):
    return clust1.distance(clust2)

vect_get_dist = np.vectorize(get_dist)




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