from typing import List, Tuple
from numpy.typing import NDArray
from collections import Counter
from abc import ABC
import numpy as np


class DataPoint:
    def __init__(self, data: NDArray[np.floating], classification: int):
        self.data = data
        self.std_data = (data - data.mean()) / data.std()
        self.classification = classification

    def __repr__(self) -> str:
        return f"Data: {self.data}; Classification: {self.classification}"

    def distance(self, other) -> np.floating:
        return np.linalg.norm(self.data - other.data)

    def standard_distance(self, other) -> np.floating:
        return np.linalg.norm(self.std_data - other.std_data)


class KNN(ABC):
    def __init__(self, k: int, data: NDArray[np.floating]):
        self.datapoints = [DataPoint(row[:-1], row[-1]) for row in data]
        self.k = k

    def test(self, row: NDArray[np.floating]) -> Tuple[int, int]:
        expected = row[-1]
        datapoint = DataPoint(row[:-1], expected)

        distances = self.get_distances(self.datapoints, datapoint)

        distances.sort(key=lambda t: t[1])
        neighbors = distances[: self.k]

        actual = self.estimate_class(neighbors)

        return expected, actual

    @staticmethod
    def estimate_class(neighbors: List[Tuple[int, np.floating]]) -> int:
        raise

    @staticmethod
    def get_distances(
        datapoints: List[DataPoint], datapoint: DataPoint
    ) -> List[Tuple[int, np.floating]]:
        raise


class BasicKNN(KNN):
    @staticmethod
    def estimate_class(neighbors: List[Tuple[int, np.floating]]) -> int:
        ns = [dist[0] for dist in neighbors]
        return Counter(ns).most_common()[0][0]

    @staticmethod
    def get_distances(
        datapoints: List[DataPoint], datapoint: DataPoint
    ) -> List[Tuple[int, np.floating]]:
        return [(p.classification, p.distance(datapoint)) for p in datapoints]

    def __repr__(self) -> str:
        return "Basic KNN"


class WeightedKNN(KNN):
    @staticmethod
    def estimate_class(neighbors: List[Tuple[int, np.floating]]) -> int:
        counts = {}
        zeros = []
        for n, distance in neighbors:
            if distance == 0:
                zeros.append(n)
                continue

            weight = 1 / (distance**2)

            if n not in counts:
                counts[n] = 0
            counts[n] += weight

        match len(zeros):
            case 0:
                return max(zip(counts.values(), counts.keys()))[1]
            case 1:
                return zeros[0]
            case _:
                return Counter(zeros).most_common()[0][0]

    @staticmethod
    def get_distances(
        datapoints: List[DataPoint], datapoint: DataPoint
    ) -> List[Tuple[int, np.floating]]:
        # NOTE: standard_distance is not in use because it gives worse results (weird)
        # return [(p.classification, p.standard_distance(datapoint)) for p in datapoints]
        return [(p.classification, p.distance(datapoint)) for p in datapoints]

    def __repr__(self) -> str:
        return "Weighted distances KNN"
