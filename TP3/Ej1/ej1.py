import numpy as np
from perceptron import Perceptron
from svm import SVM
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

def plot_points(data, w, name=None):
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
    x = np.linspace(0, 10, 100)
    y = -(w[2] + w[0]*x) / w[1]

    line, = plt.plot(x, y, 'g-')

    return line, fig

def plot_points_gif(data, weights, name):
    line, fig = plot_points(data, weights[0])

    x = np.linspace(0, 10, 100)

    def update(i):
        if i + 1 < len(weights):
            w = weights[i + 1]

            y = -(w[2] + w[0] * x) / w[1]
            
            line.set_ydata(y)

    animation = ani.FuncAnimation(fig, update, frames=len(weights) + 100) # type: ignore
    animation.save(f"./Out/animation_{name}.gif", writer="imagemagick", fps=30)

def calculate_margins(p1, p2, p, X, y): 
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - m * p1[0]

    d = np.abs(m*p[0] + b - p[1]) / np.sqrt(1 + m**2)

    tv = (d * m / np.sqrt(m**2 + 1), d * -1 / np.sqrt(m**2 + 1))

    b += (tv[0] * m - tv[1]) / 2

    r = ((X[:,0] * m + b - X[:,1]) * y)

    if np.any(r < 0):
        return None
    return (min(r), m, b)

def main():
    dim = 2
    total = 20

    TP3_1 = generate_classified(total, dim)
    X = TP3_1[:, :-1]
    y = TP3_1[:, -1] 

    perceptron = Perceptron(X, y, EPOCHS, LEARNING_RATE)

    weights = perceptron.train()
    
    w = weights[-1][:-1]
    b = weights[-1][-1]

    print(f"W: {w}\nB: {b}\n")

    plot_points(TP3_1, [w[0], w[1], b])

    # NOTE: Maximize margins

    results = np.dot(X, w) + b
    indices = np.argpartition(np.abs(results), 10)[:10]

    i1 = indices[results[indices] > 0]
    i2 = indices[results[indices] < 0]

    c1 = X[i1]
    c2 = X[i2]

    if len(c1) == 0 or len(c2) == 0:
        raise Exception("BadBatch")

    max_margin = None

    for i in range(len(c1)):
        p1 = c1[i]
        for j in range(i + 1, len(c1)):
            p2 = c1[j]
            for k in range(len(c2)):
                p = c2[k]

                margin = calculate_margins(p1, p2, p, X, y)
                if margin is None:
                    continue

                if max_margin is None or margin[0] > max_margin[0]:
                    max_margin = margin


    for i in range(len(c2)):
        p1 = c2[i]
        for j in range(i + 1, len(c2)):
            p2 = c2[j]
            for k in range(len(c1)):
                p = c1[k]

                margin = calculate_margins(p1, p2, p, X, y)
                if margin is None:
                    continue

                if max_margin is None or margin[0] > max_margin[0]:
                    max_margin = margin

    if max_margin is None:
        raise Exception("Bad batch")
    plot_points(TP3_1, [max_margin[1], -1, max_margin[2]])

    # plot_points_gif(TP3_1, weights, "TP3_1")

    svm = SVM(X, y, EPOCHS*30, LEARNING_RATE,100)

    (weights, best) = svm.train()

    i = best[1]

    w = weights[i][:-1]
    b = weights[i][-1]

    print(f"W: {w}\nB: {b}\n")

    plot_points(TP3_1, [w[0], w[1], b])
    plt.show()

    # plot_points_gif(TP3_1, weights, "TP3_1-svm")
    
    # --------------------------------------------------------------------------------------

    TP3_2 = generate_classified(total, dim, miss_some=True)
    X = TP3_1[:, :-1]
    y = TP3_1[:, -1] 

    perceptron = Perceptron(X, y, EPOCHS, LEARNING_RATE)

    w = perceptron.train()
    print(f"W: {w[-1][:-1]}\nB: {w[-1][-1]}\n")

    plot_points_gif(TP3_2, w, "TP3_2")


if __name__ == "__main__":
    main()
