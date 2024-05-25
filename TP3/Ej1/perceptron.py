import random
import matplotlib.pyplot as plt
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
        self.__plot_recta((-self.w[1], self.w[0]))
        p = self.X.shape[0] 

        for epoch in range(self.epochs):
            for i in range(p):
                h = np.dot(self.X[i], self.w) + self.b
                activation = np.sign(h)

                self.w += self.learning_rate * (self.y[i] - activation) * self.X[i]
                self.b += self.learning_rate * (self.y[i] - activation)

                error = self.__calculate_error()

            self.__plot_recta((-self.w[1], self.w[0]))

            if error == 0:
                break

        return self.w, self.b
    
    def __calculate_error(self):
        linear_output = np.dot(self.X, self.w) + self.b
        activation = np.sign(linear_output)
        error = np.sum(activation != self.y)
        return error

    def __plot_recta(self, vector):
        x = np.linspace(-5, 5, 100)
        if vector[0] != 0:  
            y = -vector[1]/vector[0] * x - self.b/vector[0]
            if self.line is None:
                line, = plt.plot(x, y, 'g-')  
                self.line = line
            else:
                self.line.set_ydata(y)
            plt.legend(loc='upper left')
            plt.draw()
            plt.pause(0.1)
