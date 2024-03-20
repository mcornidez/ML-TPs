import statistics
import typing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def simple_linear_regresion(x, y, x_label="...", y_label="Sales", title="..."):
    cov_xy = statistics.covariance(x, y)
    var_x = statistics.variance(x)
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)

    b = cov_xy / var_x

    a = mean_y - b * mean_x

    predicted_y = a + x * b

    totalSquareSum = sum((y - mean_y) ** 2)
    modelSquareSum = sum((predicted_y - mean_y) ** 2)
    resudueSquareSum = sum((y - predicted_y) ** 2)

    r2 = modelSquareSum / totalSquareSum

    print(f"R2 = {r2}")
    print(f"Is close: {np.isclose(statistics.covariance(predicted_y, y - predicted_y), 0)}")

    # NOTE: Draw data
    plt.scatter(x, y)

    # NOTE: Draw line
    plt.plot(x, predicted_y, color="red", label=f"y = {a:.3f} + {b:.3f}x")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    plt.show()

    return (a, b)


def multiple_linear_regresion(x, y):
    cov_xy = np.array([statistics.covariance(xi, y) for xi in x])
    var_x = np.var(x, axis=1, ddof = 1)

    mean_x = x.mean(axis=1)
    mean_y = y.mean()

    print(statistics.covariance(x[0], x[1]))
    print(statistics.covariance(x[1], x[2]))
    print(statistics.covariance(x[0], x[2]))

    b = cov_xy / var_x

    a = mean_y - np.dot(b, mean_x)

    predicted_y = a + np.dot(np.transpose(x), b)

    totalSquareSum = sum((y - mean_y) ** 2)
    modelSquareSum = sum((predicted_y - mean_y) ** 2)
    resudueSquareSum = sum((y - predicted_y) ** 2)

    r2 = modelSquareSum / totalSquareSum
    print(1 - resudueSquareSum / totalSquareSum)
    #Para que se de la igualdad esto de abajo deberia ser cercado a 0, como pasa en los casos unidimensionales
    print(f"Is close: {np.isclose(statistics.covariance(predicted_y, y - predicted_y), 0)}")

    print(f"R2 = {r2}")

    return (a, b)


def main():
    None
    df = pd.read_csv("Advertising.csv", index_col=0)

    sales = df["Sales"].to_numpy()
    tv = df["TV"].to_numpy()
    radio = df["Radio"].to_numpy()
    newspaper = df["Newspaper"].to_numpy()

    (a_tv, b_tv) = simple_linear_regresion(tv, sales, title="Sales vs TV", x_label="TV")
    (a_radio, b_radio) = simple_linear_regresion(
        radio, sales, title="Sales vs Radio", x_label="Radio"
    )
    (a_newspaper, b_newspaper) = simple_linear_regresion(
        newspaper, sales, title="Sales vs Newspaper", x_label="Newspaper"
    )

    (a_all, b_all) = multiple_linear_regresion(np.array([tv, radio, newspaper]), sales)


if __name__ == "__main__":
    main()
