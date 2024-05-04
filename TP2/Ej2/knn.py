import numpy as np
from collections import Counter


class DataPoint:
    def __init__(self, data, classification):
        self.data = data
        self.classification = classification

    def __repr__(self) -> str:
        return f"Data: {self.data}; Classification: {self.classification}"

    def distance(self, other):
        return np.linalg.norm(self.data - other.data)


class KNN:
    def __init__(self, k, data):
        self.datapoints = [DataPoint(row[:-1], row[-1]) for row in data]
        self.k = k

    def test(self, row):
        expected = row[-1]
        datapoint = DataPoint(row[:-1], None)

        distances = [(point, point.distance(datapoint)) for point in self.datapoints]

        distances.sort(key=lambda t: t[1])

        neighbors = [dist[0].classification for dist in distances[: self.k]]

        actual = Counter(neighbors).most_common()[0][0]

        return expected, actual
