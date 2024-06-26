from typing import List, Tuple
from numpy.typing import NDArray
from collections import Counter
from abc import ABC
import numpy as np


class DataPoint:
    def __init__(self, data: NDArray[np.floating], classification: int):
        self.data = data
        self.classification = classification

    def __repr__(self) -> str:
        return f"Data: {self.data}; Classification: {self.classification}"

    def distance(self, other) -> np.floating:
        return np.linalg.norm(self.data - other.data)


class KNN(ABC):
    def __init__(self, k: int, data: NDArray[np.floating]):
        self.datapoints = [DataPoint(row[:-1], row[-1]) for row in data]
        self.k = k

    def test(self, row: NDArray[np.floating]) -> Tuple[int, int]:
        expected = row[-1]
        datapoint = DataPoint(row[:-1], expected)

        distances = [(p.classification, p.distance(datapoint)) for p in self.datapoints]

        distances.sort(key=lambda t: t[1])
        neighbors = distances[: self.k]

        actual = self.estimate_class(neighbors)

        return expected, actual

    @staticmethod
    def estimate_class(neighbors: List[Tuple[int, np.floating]]) -> int:
        raise


class BasicKNN(KNN):
    @staticmethod
    def estimate_class(neighbors: List[Tuple[int, np.floating]]) -> int:
        counts = {}
        for c, distance in neighbors:
            if c not in counts:
                counts[c] = [0, 0]

            counts[c][0] += 1
            # NOTE: Use the distance to keep the nearest in tie
            counts[c][1] -= distance

        return max(zip(counts.values(), counts.keys()))[1]

    def __repr__(self) -> str:
        return "Basic KNN"


class WeightedKNN(KNN):
    @staticmethod
    def estimate_class(neighbors: List[Tuple[int, np.floating]]) -> int:
        counts = {}
        zeros = []
        for c, distance in neighbors:
            if distance == 0:
                zeros.append(c)
                continue

            weight = 1 / (distance**2)

            if c not in counts:
                counts[c] = [0, 0]

            counts[c][0] += weight
            # NOTE: Use the distance to keep the nearest in tie
            counts[c][1] -= distance

        match len(zeros):
            case 0:
                return max(zip(counts.values(), counts.keys()))[1]
            case 1:
                return zeros[0]
            case _:
                return Counter(zeros).most_common()[0][0]

    def __repr__(self) -> str:
        return "Weighted distances KNN"
