import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from metrics import Confusion
from utils import wordcount_one_stars_mean, prepare_data
from knn import BasicKNN, WeightedKNN

os.makedirs("./Out", exist_ok=True)

CATEGORIES = [1, 2, 3, 4, 5]

rng = np.random.default_rng()


def divide_data(data):
    sample = rng.choice(data, replace=True, size=len(data))

    PERCENTAGE = 0.8

    divider = int(len(sample) * PERCENTAGE)

    train = sample[:divider]
    test = sample[divider:]

    return train, test


def run_knn(data, knn_gen, k_start, k_end, runs, standarized=False):
    runs_for_k = {}

    print("Progress (k):")
    for k in range(k_start, k_end + 1, 2):
        print(k, end=" ", flush=True)
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
    print()

    return runs_for_k


def plot_confusions(data, knn_gen):
    train, test = divide_data(data)

    for k in [1, 5, 15, 25, 49, 75, 99]:
        knn = knn_gen(k, train)
        confusion = Confusion(CATEGORIES)

        for row in test:
            expected, actual = knn.test(row)
            confusion.add_run(expected, actual)

        confusion.plot(f"{str(knn)} for K = {k}")

    for i in range(3):
        train_col = train[:, i]
        test_col = test[:, i]
        train[:, i] = (train_col - train_col.mean()) / train_col.std()
        test[:, i] = (test_col - test_col.mean()) / test_col.std()

    for k in [1, 5, 15, 25, 49, 75, 99]:
        knn = knn_gen(k, train)
        confusion = Confusion(CATEGORIES)

        for row in test:
            expected, actual = knn.test(row)
            confusion.add_run(expected, actual)

        confusion.plot(f"{str(knn)} for K = {k} standarized")


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

    plt.savefig(f"./Out/{name}.png")


def plot_datapoints(datapoints, standarized=False):
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

    plt.savefig(f"./Out/3D{' standarized' if standarized else ''}.png")


def main():
    df = pd.read_csv("../Data/reviews_sentiment.csv", sep=";")

    mean = wordcount_one_stars_mean(df)
    print(f"Mean Wordcount of 1 star review comments is {mean:.3f}")

    data = prepare_data(df)

    plot_confusions(data, BasicKNN)
    plot_confusions(data, WeightedKNN)

    print("Running basic KNN for k between 1 and 99")
    runs = run_knn(data, BasicKNN, 1, 99, 20)

    plot_precision(runs, "Basic KNN precision")

    print("Running weighted distances KNN for k between 1 and 99")
    runs = run_knn(data, WeightedKNN, 1, 99, 20)

    plot_precision(runs, "Weighted KNN precision")

    plot_datapoints(data)

    # NOTE: standarize data and run again
    for i in range(3):
        col = data[:, i]
        data[:, i] = (col - col.mean()) / col.std()

    print("Running basic KNN with standarized data for k between 1 and 99")
    runs = run_knn(data, BasicKNN, 1, 99, 20, standarized=True)

    plot_precision(runs, "Basic KNN precision standarized")

    print("Running weighted distances KNN with standarized data for k between 1 and 99")
    runs = run_knn(data, WeightedKNN, 1, 99, 20, standarized=True)

    plot_precision(runs, "Weighted KNN precision standarized")

    plot_datapoints(data, standarized=True)


if __name__ == "__main__":
    main()
