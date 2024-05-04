import pandas as pd
import numpy as np
from collections import Counter
import sys


def wordcount_one_stars_mean(df):
    stars = df[["Star Rating", "wordcount"]].to_numpy()
    one_stars = stars[stars[:, 0] == 1]
    wordcount_one_stars = one_stars[:, 1]
    return np.mean(wordcount_one_stars)


def prepare_data(df):
    PERCENTAGE = 0.8

    def sentiment_to_int(sentiment):
        if sentiment == "positive":
            return 1
        elif sentiment == "negative":
            return 0
        else:
            return None

    def map_sentiment(sentiment, backup):
        sentiment = sentiment_to_int(sentiment)

        if sentiment is not None:
            return sentiment

        return sentiment_to_int(backup)

    data = df[
        ["wordcount", "titleSentiment", "sentimentValue", "Star Rating"]
    ].to_numpy()

    # Map positive to 1, negative to 0 and NaN to value from textSentiment
    data[:, 1] = list(
        map(
            map_sentiment,
            *df[["titleSentiment", "textSentiment"]].to_numpy().T,
        )
    )

    divider = int(len(data) * PERCENTAGE)

    train = data[:divider]
    test = data[divider:]

    return train, test


class DataPoint:
    def __init__(self, data, classification):
        self.data = data
        self.classification = classification

    def __repr__(self) -> str:
        return f"Data: {self.data}; Classification: {self.classification}"

    def distance(self, other):
        return np.linalg.norm(self.data - other.data)


class KNN:
    def __init__(self, k, datapoints):
        self.k = k
        self.datapoints = datapoints

    def test(self, datapoint):
        distances = [(point, point.distance(datapoint)) for point in self.datapoints]

        distances.sort(key=lambda t: t[1])

        neighbors = [dist[0].classification for dist in distances[: self.k]]

        classification = Counter(neighbors).most_common()[0][0]

        return DataPoint(datapoint.data, classification)


def main(k):
    df = pd.read_csv("../Data/reviews_sentiment.csv", sep=";")

    mean = wordcount_one_stars_mean(df)
    print(f"Mean Wordcount of 1 star review comments is {mean:.3f}")

    train, test = prepare_data(df)

    datapoints = [DataPoint(row[:3], row[3]) for row in train]
    knn = KNN(k, datapoints)

    yay = 0
    nay = 0

    for row in test:
        classification = row[3]

        point = DataPoint(row[:3], None)
        point = knn.test(point)

        if classification == point.classification:
            yay += 1
        else:
            nay += 1

    print(f"YAY: {yay}, NAY: {nay}")


if __name__ == "__main__":
    k = int(sys.argv[1])
    main(k)
