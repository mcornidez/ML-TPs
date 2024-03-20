import statistics
import typing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def variable_correlation(x):
    cov = np.cov(x)
    std = x.std(axis=1)

    print(f"Corr TV - Radio: {cov[0,1]/(std[0]*std[1])} (Baja)")
    print(f"Corr TV - Newspaper: {cov[0,2]/(std[0]*std[2])} (Baja)")
    print(f"Corr Radio - Newspaper: {cov[1,2]/(std[1]*std[2])} (Media)")


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
    print(
        f"Is close: {np.isclose(statistics.covariance(predicted_y, y - predicted_y), 0)}"
    )

    # NOTE: Draw data
    plt.scatter(x, y)

    # NOTE: Draw line
    plt.plot(x, predicted_y, color="red", label=f"y = {a:.3f} + {b:.3f}x")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()

    plt.scatter(predicted_y, y - predicted_y)
    plt.hlines(
        y=[0],
        xmin=[predicted_y.min()],
        xmax=[predicted_y.max()],
        colors="grey",
        linestyles="--",
    )
    plt.xlabel("Predicted " + y_label)
    plt.ylabel("Residual")
    plt.show()

    return (a, b)


def multiple_linear_regresion(x, y):
    # cov_xy = np.array([statistics.covariance(xi, y) for xi in x])
    # var_x = x.var(axis=1, ddof=1)

    # b = cov_xy / var_x

    # a = mean_y - np.dot(b, mean_x)

    # predicted_y = a + np.dot(x.transpose(), b)

    x = np.append(np.ones((1, x.shape[1])), x, axis=0).T

    b = np.linalg.inv(x.T @ x) @ x.T @ y

    predicted_y = x @ b

    mean_x = x.mean(axis=1)
    mean_y = y.mean()

    totalSquareSum = sum((y - mean_y) ** 2)
    modelSquareSum = sum((predicted_y - mean_y) ** 2)
    resudueSquareSum = sum((y - predicted_y) ** 2)

    r2 = modelSquareSum / totalSquareSum
    # Para que se de la igualdad esto de abajo deberia ser cercado a 0, como pasa en los casos unidimensionales
    # Explicado en respuesta a este post
    # https://stats.stackexchange.com/questions/265869/confused-with-residual-sum-of-squares-and-total-sum-of-squares
    print(
        f"Is close: {np.isclose(statistics.covariance(predicted_y, y - predicted_y), 0)}"
    )

    print(f"R2 = {r2}")
    print(f"R2' = {1 - resudueSquareSum / totalSquareSum}")

    plt.scatter(predicted_y, y - predicted_y)
    plt.hlines(
        y=[0],
        xmin=[predicted_y.min()],
        xmax=[predicted_y.max()],
        colors="grey",
        linestyles="--",
    )
    plt.xlabel("Predicted Sales")
    plt.ylabel("Residual")
    plt.show()

    return b


def main():
    None
    df = pd.read_csv("Advertising.csv", index_col=0)

    sales = df["Sales"].to_numpy()
    tv = df["TV"].to_numpy()
    radio = df["Radio"].to_numpy()
    newspaper = df["Newspaper"].to_numpy()
    x = np.array([tv, radio, newspaper])

    variable_correlation(x)
    (a_tv, b_tv) = simple_linear_regresion(tv, sales, title="Sales vs TV", x_label="TV")
    (a_radio, b_radio) = simple_linear_regresion(
        radio, sales, title="Sales vs Radio", x_label="Radio"
    )
    (a_newspaper, b_newspaper) = simple_linear_regresion(
        newspaper, sales, title="Sales vs Newspaper", x_label="Newspaper"
    )

    bs_all = multiple_linear_regresion(x, sales)
    a_all = bs_all[0]
    bs_all = bs_all[1:]


if __name__ == "__main__":
    main()
