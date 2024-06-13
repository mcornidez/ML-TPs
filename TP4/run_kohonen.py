from kohonen import KohonenNetwork
import numpy as np
from utils import get_data
import plots

def main():
    data, df = get_data()

    # data = data[:500]

    size = int(data.shape[1] / 2) # ni idea que poner, mas tarda demasiado
    radius = None
    learning_rate = None
    weights = None
    dim = data.shape[1]

    kohonen = KohonenNetwork(size, radius, learning_rate, weights, dim)

    kohonen.train(data)

    hit_matrix = np.zeros((size, size))
    genre_matrix_list = [[set([]) for j in range(size)] for i in range(size)]

    for idx, element in enumerate(data):
        i, j = kohonen.get_closest_weight_to_element_index(element)
        hit_matrix[i, j] += 1
        genre_matrix_list[i][j].add(df["genres"][idx])

    plots.heatmap(hit_matrix, "hits.png", text=hit_matrix)

    genre_matrix = [["" for j in range(size)] for i in range(size)]
    for i in range(size):
        for j in range(size):
            genre_matrix[i][j] = "\n".join(genre_matrix_list[i][j])

    plots.heatmap(hit_matrix, "genres.png", text=genre_matrix)

    u_matrix = np.zeros((size, size))
    neighbor_indices = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(size):
        for j in range(size):
            distances = []
            for ni, nj in neighbor_indices:
                if (
                    0 <= i + ni < size and 0 <= j + nj < size
                ):  # Check if neighbor is within bounds
                    distances.append(
                        np.linalg.norm(
                            kohonen.matrix[i, j] - kohonen.matrix[i + ni, j + nj]
                        )
                    )
            u_matrix[i, j] = np.mean(distances)

    plots.heatmap(u_matrix, "umatrix.png", "u-matrix")


if __name__ == "__main__":
    main()
