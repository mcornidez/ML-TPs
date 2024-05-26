import numpy as np


def loss(t):
    return np.where(t < 1, 1 - t, 0)


class SVM:
    def __init__(self, X, y, epochs, learning_rate, C=1.0) -> None:
        self.X = X
        self.y = y
        self.b = 0
        self.line = None
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.C = C

    def train(self):
        self.w = np.random.rand(self.X.shape[1])
        self.b = np.random.rand()
        weights = [np.append(self.w, self.b)]
        p = self.X.shape[0]

        best_cost = (self.cost(), 0)

        for epoch in range(1, self.epochs):
            learning_rate = self.learning_rate * np.exp(-0.0001 * epoch)

            i = np.random.randint(0, p)

            t = (np.dot(self.X[i], self.w) + self.b) * self.y[i]

            if t < 1:
                self.w += self.learning_rate * (self.C * self.X[i] * self.y[i] - self.w)
                self.b += self.learning_rate * self.C * self.y[i]
            else:
                self.w -= learning_rate * self.w

            weights.append(np.append(self.w, self.b))

            cost = (self.cost(), epoch)

            if cost[0] < best_cost[0]:
                best_cost = cost

        return (weights, best_cost)

    def cost(self):
        t = (np.dot(self.X, self.w) + self.b) * self.y

        return (np.linalg.norm(self.w) ** 2) / 2 + self.C * np.sum(loss(t))
