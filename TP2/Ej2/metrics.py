from typing import Any, List
import numpy as np
from numpy.typing import NDArray


class Metrics:
    # 0 -> TP, 1 -> TN, 2 -> FP, 3 -> FN
    def __init__(self, metrics: NDArray[np.floating]):
        self.TP = metrics[0]
        self.TN = metrics[1]
        self.FP = metrics[2]
        self.FN = metrics[3]

    def precision(self):
        return self.TP / (self.TP + self.FP)

    def accuracy(self):
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)

    def TVP(self):
        return self.TP / (self.TP + self.FN)

    def TFP(self):
        return self.FP / (self.FP + self.TN)

    def F1(self):
        return (2 * self.precision() * self.TVP()) / (self.precision() + self.TVP())


class Confusion:
    def __init__(self, categories: List[Any]):
        self.categories = categories
        self.size = len(categories)
        self.matrix = np.zeros((self.size, self.size))
        self.stats = None

    def add_run(self, expected, actual):
        i = self.categories.index(expected)
        j = self.categories.index(actual)
        self.matrix[i, j] += 1

    def metrics(self) -> Metrics:
        if self.stats is not None:
            return self.stats

        metrics = np.zeros((self.size, 4))

        # 0 -> TP, 1 -> TN, 2 -> FP, 3 -> FN
        for i in range(self.size):
            metrics[i][0] = self.matrix[i][i]
            metrics[i][2] = self.matrix.transpose()[i].sum() - metrics[i][0]
            metrics[i][3] = self.matrix[i].sum() - metrics[i][0]
            metrics[i][1] = (
                self.matrix.sum().sum() - metrics[i][0] - metrics[i][3] - metrics[i][2]
            )
        self.stats = Metrics(metrics.transpose())

        return self.stats
