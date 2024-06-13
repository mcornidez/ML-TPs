import numpy as np
from numpy import ndarray


class KohonenNetwork:
    def __init__(
        self,
        size: int,
        radius: float | None = None,
        learning_rate: float | None = None,
        weights: ndarray | None = None,
        dim: int | None = None,
    ):
        self._size = size

        if learning_rate is None:
            self._learning_rate = 1
            self._update_learning_rate = True
        else:
            assert learning_rate < 1
            self._learning_rate = learning_rate
            self._update_learning_rate = False

        if radius is None:
            self._radius = size
            self._update_radius = True
        else:
            assert radius >= 1
            self._radius = radius
            self._update_radius = False

        if weights is not None:
            total = size * size

            assert weights.shape[0] > total

            matrix = np.zeros((size, size, weights.shape[1]))
            indices = np.random.choice(weights.shape[0], size=total, replace=False)

            for i in range(size):
                for j in range(size):
                    index = indices[
                        i * size + j
                    ]  # choose the next random index from the list
                    matrix[i, j] = weights[index]
        else:
            assert dim is not None
            matrix = np.random.uniform(-1, 1, size=(size, size, dim))

        self._matrix = matrix

    @property
    def matrix(self) -> ndarray:
        return self._matrix

    def train(self, data: ndarray):
        assert data.shape[1] == self._matrix.shape[2]

        for epoch in range(1, 500):
            self.update_parameters(epoch)
            np.random.shuffle(data)
            for element in data:
                i_win, j_win = self.get_closest_weight_to_element_index(element)

                for i in range(self._size):
                    for j in range(self._size):
                        d = np.sqrt((i - i_win) ** 2 + (j - j_win) ** 2)
                        V = np.exp(-2 * d / self._radius)
                        self._matrix[i, j] += (
                            self._learning_rate * V * (element - self._matrix[i, j])
                        )

            if epoch % 10 == 0:
                print(epoch)

    def update_parameters(self, epoch: int):
        if self._update_learning_rate:
            self._learning_rate = 1 / epoch

        if self._update_radius:
            self._radius = self._size / epoch

    def get_closest_weight_to_element_index(self, value: ndarray) -> tuple[int, int]:
        # compute the Euclidean distances between value and all the vectors in the matrix
        distances = np.linalg.norm(self._matrix - value, axis=2)
        return np.unravel_index(distances.argmin(), distances.shape)[:2]

    def u_matrix(self):
        u_matrix = np.zeros((self._size, self._size))
        neighbor_indices = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for i in range(self._size):
            for j in range(self._size):
                distances = []
                for ni, nj in neighbor_indices:
                    if (
                        0 <= i + ni < self._size and 0 <= j + nj < self._size
                    ):  # Check if neighbor is within bounds
                        distances.append(
                            np.linalg.norm(
                                self.matrix[i, j] - self.matrix[i + ni, j + nj]
                            )
                        )
                u_matrix[i, j] = np.mean(distances)

        return u_matrix
