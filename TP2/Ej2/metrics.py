from typing import Any, List
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


class Metrics:
    # 0 -> TP, 1 -> TN, 2 -> FP, 3 -> FN
    def __init__(self, metrics: NDArray[np.floating]):
        self.TP = metrics[0]
        self.TN = metrics[1]
        self.FP = metrics[2]
        self.FN = metrics[3]

    def precision(self):
        a = self.TP
        b = self.TP + self.FP
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

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

    def plot(self, name):
        plt.figure()
        plt.imshow(self.matrix, cmap="Blues", interpolation="nearest")
        plt.title(name)
        plt.colorbar()

        plt.xlabel("Predicted")
        plt.ylabel("Real")
        plt.xticks(
            np.arange(self.matrix.shape[1]),
            list(map(str, range(1, self.matrix.shape[1] + 1))),
        )
        plt.yticks(
            np.arange(self.matrix.shape[0]),
            list(map(str, range(1, self.matrix.shape[0] + 1))),
        )

        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                plt.text(
                    j,
                    i,
                    str(int(self.matrix[i, j])),
                    ha="center",
                    va="center",
                    color="black",
                )

        plt.savefig(f"../Out/{name}.png")
