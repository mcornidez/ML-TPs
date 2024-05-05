import pandas as pd
import sys
from metrics import Confusion
from utils import wordcount_one_stars_mean, prepare_data
from knn import KNN, BasicKNN, WeightedKNN

CATEGORIES = [1, 2, 3, 4, 5]


def main(k):
    df = pd.read_csv("../Data/reviews_sentiment.csv", sep=";")

    mean = wordcount_one_stars_mean(df)
    print(f"Mean Wordcount of 1 star review comments is {mean:.3f}")

    train, test = prepare_data(df)

    runs_for_k = {}

    for k in range(1, 50):
        runs_for_k[k] = []

        for _ in range(1, 10):
            knn = BasicKNN(k, train)
            confusion = Confusion(CATEGORIES)

            for row in test:
                expected, actual = knn.test(row)
                confusion.add_run(expected, actual)

            metrics = confusion.metrics()
            runs_for_k[k].append(metrics.precision())

        # print(confusion.matrix)


if __name__ == "__main__":
    k = int(sys.argv[1])
    main(k)
