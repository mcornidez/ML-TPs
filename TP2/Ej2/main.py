import pandas as pd
import sys
from utils import wordcount_one_stars_mean, prepare_data
from knn import KNN, BasicKNN, WeightedKNN


def run_knn(knn: KNN, test):
    yay = 0
    nay = 0

    for row in test:
        expected, actual = knn.test(row)

        if expected == actual:
            yay += 1
        else:
            nay += 1

    print(f"YAY: {yay}, NAY: {nay}")


def main(k):
    df = pd.read_csv("../Data/reviews_sentiment.csv", sep=";")

    mean = wordcount_one_stars_mean(df)
    print(f"Mean Wordcount of 1 star review comments is {mean:.3f}")

    train, test = prepare_data(df)

    print(f"Running basic knn with k = {k}")

    run_knn(BasicKNN(k, train), test)

    print(f"Running weighted knn with k = {k}")

    run_knn(WeightedKNN(k, train), test)


if __name__ == "__main__":
    k = int(sys.argv[1])
    main(k)
