import numpy as np
import pandas as pd
from perceptron import Perceptron
import matplotlib.pyplot as plt

EPOCHS = 5000
LEARNING_RATE= 0.01

def generate_classified(amount, dim, miss_some = False):
    classifier = [np.random.uniform(5,10), -np.random.uniform(5,10), np.random.uniform(5,10)]

    points = np.random.uniform(0, 10, (amount, dim))

    with_ones = np.concatenate((points, np.ones((amount, 1))), axis=1)

    classification = [np.dot(row, classifier) for row in with_ones]

    classified_points = np.concatenate((points, np.array(classification).reshape((amount, 1))), axis=1)

    if miss_some:
        classified_points = np.array(sorted(classified_points, key=lambda row: np.abs(row[-1])))

        classified_points[3, -1] *= -1
        classified_points[4, -1] *= -1

        np.random.shuffle(classified_points)

    classified_points[:,-1] = np.sign(classified_points[:,-1])

    return classified_points


def main():
    dim = 2
    total = 20

    TP3_1 = generate_classified(total, dim)
    X = TP3_1[:, :-1]
    y = TP3_1[:, -1] 

    plt.figure()
    plt.gca().set_aspect('equal', adjustable='box')
    df_cat1 = TP3_1[TP3_1[:, -1] == 1]
    df_cat2 = TP3_1[TP3_1[:, -1] == -1]
    plt.scatter(df_cat1[:, 0], df_cat1[:, 1], c='r', label='Clase 1')
    plt.scatter(df_cat2[:, 0], df_cat2[:, 1], c='b', label='Clase -1')
    plt.legend(loc='upper left')
    plt.xlim((0, 10))
    plt.ylim((0, 10))
    plt.title('Recta de separacion')
    plt.grid(True)
    plt.legend(loc='upper left')

    perceptron = Perceptron(X, y, EPOCHS, LEARNING_RATE)

    w, b = perceptron.train()
    plt.show()
    print(f"W: {w}\nB: {b}\n")

if __name__ == "__main__":
    main()
