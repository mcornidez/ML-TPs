import numpy as np
import pandas as pd
from perceptron import Perceptron
import matplotlib.pyplot as plt

EPOCHS = 5000
LEARNING_RATE= 0.01

def generate_classified(amount, dim, vec_classifier):
    points = np.random.uniform(-1, 1, (amount, dim))
    classified_points = np.concatenate((points, vec_classifier(points[:, dim-1]).reshape((amount, 1))), axis=1) 
    return classified_points

def main():
    dim = 2
    total = 10
    classif = 8
    misclassif = total - classif

    TP3_1 = generate_classified(total, dim, np.sign)
    X = TP3_1[:, :-1]
    y = TP3_1[:, -1] 

    plt.figure()
    plt.gca().set_aspect('equal', adjustable='box')
    df_cat1 = TP3_1[TP3_1[:, -1] == 1]
    df_cat2 = TP3_1[TP3_1[:, -1] == -1]
    plt.scatter(df_cat1[:, 0], df_cat1[:, 1], c='r', label='Clase 1')
    plt.scatter(df_cat2[:, 0], df_cat2[:, 1], c='b', label='Clase -1')
    plt.legend(loc='upper left')
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.title('Recta de separacion')
    plt.grid(True)
    plt.legend(loc='upper left')

    perceptron = Perceptron(X, y, EPOCHS, LEARNING_RATE)

    w, b = perceptron.train()
    plt.show()
    print(f"W: {w}\nB: {b}\n")

if __name__ == "__main__":
    main()
