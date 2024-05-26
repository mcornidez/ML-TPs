import numpy as np


class Perceptron:
    def __init__(self, X, y, epochs, learning_rate) -> None:
        self.X = X
        self.y = y
        self.b = 0
        self.line = None
        self.epochs = epochs
        self.learning_rate = learning_rate

    def train(self):
        self.w = np.random.rand(self.X.shape[1])
        self.b = np.random.rand()
        weights = [np.append(self.w, self.b)]
        p = self.X.shape[0]
        error = 0

        for epoch in range(self.epochs):
            # learning_rate = self.learning_rate * np.exp(-0.0001 * epoch)

            i = np.random.randint(0, p)

            h = np.dot(self.X[i], self.w) + self.b
            activation = np.sign(h)

            self.w += self.learning_rate * (self.y[i] - activation) * self.X[i]
            self.b += self.learning_rate * (self.y[i] - activation)

            error = self.calculate_error()

            weights.append(np.append(self.w, self.b))

            if error == 0:
                break

        return weights

    def calculate_error(self):
        linear_output = np.dot(self.X, self.w) + self.b
        activation = np.sign(linear_output)
        error = np.sum(activation != self.y)
        return error
