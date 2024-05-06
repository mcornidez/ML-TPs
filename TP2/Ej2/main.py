import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from metrics import Confusion
from utils import wordcount_one_stars_mean, prepare_data
from knn import BasicKNN, WeightedKNN

os.makedirs("./Out", exist_ok=True)

CATEGORIES = [1, 2, 3, 4, 5]


def divide_data(data):
    np.random.shuffle(data)

    PERCENTAGE = 0.8

    divider = int(len(data) * PERCENTAGE)

    train = data[:divider]
    test = data[divider:]

    return train, test


def run_knn(data, knn_gen, k_start, k_end, runs):
    runs_for_k = {}

    for k in range(k_start, k_end + 1, 2):
        print(k)
        runs_for_k[k] = []

        for _ in range(runs):
            train, test = divide_data(data)

            knn = knn_gen(k, train)
            confusion = Confusion(CATEGORIES)

            for row in test:
                expected, actual = knn.test(row)
                confusion.add_run(expected, actual)

            metrics = confusion.metrics()
            runs_for_k[k].append(metrics.precision())

    train, test = divide_data(data)

    for k in [1, 5, 15, 25, 49, 75, 99]:
        knn = knn_gen(k, train)
        confusion = Confusion(CATEGORIES)

        for row in test:
            expected, actual = knn.test(row)
            confusion.add_run(expected, actual)

        confusion.plot(f"{str(knn)} for K = {k}")

    return runs_for_k


def plot_precision(runs, name):
    ks = list(runs.keys())
    precisions = list(runs.values())

    means = [np.mean(prec) for prec in precisions]
    stds = [np.std(prec) for prec in precisions]

    plt.figure()
    plt.errorbar(ks, means, yerr=stds, fmt="-o")
    plt.ylim(0, 1)
    plt.xlabel("K")
    plt.ylabel("Precision")
    plt.title(name)
    plt.grid(True)

    plt.savefig(f"../Out/{name}.png")


def plot_datapoints(datapoints):
    data = {}
    for row in datapoints:
        category = row[-1]
        if category not in data:
            data[category] = []
        data[category].append(row[:-1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for category, values in data.items():
        ax.scatter(
            xs=[point[0] for point in values],
            ys=[point[1] for point in values],
            zs=[point[2] for point in values],
            label=f"{category} Star{'' if category == 1 else 's'}",
        )  # type: ignore

    ax.legend(loc="upper left")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Title Sentiment")
    ax.set_zlabel("Sentiment Value")  # type: ignore

    plt.savefig(f"../Out/3D.png")


def main():
    df = pd.read_csv("../Data/reviews_sentiment.csv", sep=";")

    mean = wordcount_one_stars_mean(df)
    print(f"Mean Wordcount of 1 star review comments is {mean:.3f}")

    data = prepare_data(df)

    print("Running basic KNN for k between 1 and 49")
    runs = run_knn(data, BasicKNN, 1, 99, 20)

    plot_precision(runs, "Basic KNN precision")

    print("Running weighted distances KNN for k between 1 and 49")
    runs = run_knn(data, WeightedKNN, 1, 99, 20)

    plot_precision(runs, "Weighted KNN precision")

    plot_datapoints(data)


if __name__ == "__main__":
    main()
