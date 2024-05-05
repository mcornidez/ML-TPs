import numpy as np


def wordcount_one_stars_mean(df):
    stars = df[["Star Rating", "wordcount"]].to_numpy()
    one_stars = stars[stars[:, 0] == 1]
    wordcount_one_stars = one_stars[:, 1]
    return np.mean(wordcount_one_stars)


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


def prepare_data(df):

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

    return data
