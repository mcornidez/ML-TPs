from re import sub
from kohonen import KohonenNetwork
import numpy as np
from utils import get_data, get_subset
import plots


def main():
    data, df = get_data()

    size = 6
    radius = None
    learning_rate = None
    weights = data
    dim = data.shape[1]

    kohonen = KohonenNetwork(size, radius, learning_rate, weights, dim)

    # kohonen.train(data)

    hit_matrix = np.zeros((size, size))

    for idx, element in enumerate(data):
        i, j = kohonen.get_closest_weight_to_element_index(element)
        hit_matrix[i, j] += 1

    plots.heatmap(hit_matrix, "hits.png", text=hit_matrix)

    u_matrix = kohonen.u_matrix()

    plots.heatmap(u_matrix, "umatrix.png", "u-matrix")

    subset, dfs = get_subset()

    kohonen = KohonenNetwork(size, radius, learning_rate, None, dim)

    kohonen.train(subset)

    hit_matrix = np.zeros((size, size))
    genre_matrix_list = [[set([]) for j in range(size)] for i in range(size)]

    for idx, element in enumerate(subset):
        i, j = kohonen.get_closest_weight_to_element_index(element)
        hit_matrix[i, j] += 1
        genre_matrix_list[i][j].add(dfs["genres"][idx])

    plots.heatmap(hit_matrix, "hit_discriminated.png", text=hit_matrix)

    genre_matrix = [["" for j in range(size)] for i in range(size)]
    for i in range(size):
        for j in range(size):
            genre_matrix[i][j] = "\n".join(genre_matrix_list[i][j])

    plots.heatmap(hit_matrix, "genres_discriminated.png", text=genre_matrix, fontSize=5)

    u_matrix = kohonen.u_matrix()

    plots.heatmap(u_matrix, "umatrix_discriminated.png", "u-matrix")

    matrixes = {}
    for idx, element in enumerate(subset):
        genre = dfs["genres"][idx]
        if genre not in matrixes:
            matrixes[genre] = np.zeros((size, size))

        i, j = kohonen.get_closest_weight_to_element_index(element)

        matrixes[genre][i, j] += 1

    for genre, matrix in matrixes.items():
        plots.heatmap(matrix, f"hit_matrix_{genre}.png", text=matrix, title=genre)


if __name__ == "__main__":
    main()
