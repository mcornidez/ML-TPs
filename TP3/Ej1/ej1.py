import numpy as np
import pandas as pd
from perceptron import Perceptron
import matplotlib.pyplot as plt
import matplotlib.animation as ani

import os
os.makedirs("./Out", exist_ok=True)

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


def plot_points(data, weights, name):
    fig = plt.figure()
    plt.gca().set_aspect('equal', adjustable='box')
    df_cat1 = data[data[:, -1] == 1]
    df_cat2 = data[data[:, -1] == -1]
    plt.scatter(df_cat1[:, 0], df_cat1[:, 1], c='r', label='Clase 1')
    plt.scatter(df_cat2[:, 0], df_cat2[:, 1], c='b', label='Clase -1')
    plt.legend(loc='upper left')
    plt.xlim((0, 10))
    plt.ylim((0, 10))
    plt.grid(True)

    w = weights[0]

    x = np.linspace(0, 10, 100)
    y = -(w[2] + w[0]*x) / w[1]

    line, = plt.plot(x, y, 'g-')

    def update(i):
        if i + 1 < len(weights):
            w = weights[i + 1]

            y = -(w[2] + w[0] * x) / w[1]
            
            line.set_ydata(y)

    animation = ani.FuncAnimation(fig, update, frames=len(weights) + 100) # type: ignore
    animation.save(f"./Out/animation_{name}.gif", writer="imagemagick", fps=30)


def main():
    dim = 2
    total = 20

    TP3_1 = generate_classified(total, dim)
    X = TP3_1[:, :-1]
    y = TP3_1[:, -1] 

    perceptron = Perceptron(X, y, EPOCHS, LEARNING_RATE)

    w = perceptron.train()
    print(f"W: {w[-1][:-1]}\nB: {w[-1][-1]}\n")

    plot_points(TP3_1, w, "TP3_1")

if __name__ == "__main__":
    main()
