from hierarchical import Method, train_hierarchical
import numpy as np
from utils import get_data, data_len
import plots
import sys
import pickle
sys.setrecursionlimit(10000)


from scipy.cluster.hierarchy import linkage


def main():
    data, df = get_data()

    # Test data
    X = np.array(
        [
            [0.40, 0.53],
            [0.22, 0.38],
            [0.35, 0.32],
            [0.26, 0.19],
            [0.08, 0.41],
            [0.45, 0.30],
        ]
    )

    #linkage_min = train_hierarchical(X, Method.MIN)
    #print("min")
    #plots.dendogram(linkage_min, "")
    # Run with library
    #linkage(data, "single")
    #plots.dendogram(linkage(data, "single"), "")
    linkage_min = train_hierarchical(X, Method.MIN)
    with open('Linkages/linkage_min_len=' + str(data_len) +'.pkl', 'wb') as file:
        pickle.dump(linkage_min, file)
    plots.dendogram(linkage_min, 'Graphs/dendr_min_len=1000.png', 'png')
    linkage_max = train_hierarchical(X, Method.MAX)
    with open('Linkages/linkage_max_len=' + str(data_len) +'.pkl', 'wb') as file:
        pickle.dump(linkage_max, file)
    plots.dendogram(linkage_min, 'Graphs/dendr_max_len=1000.png', 'png')
    linkage_avg = train_hierarchical(X, Method.AVERAGE)
    with open('Linkages/linkage_avg_len=' + str(data_len) +'.pkl', 'wb') as file:
        pickle.dump(linkage_avg, file)
    plots.dendogram(linkage_min, 'Graphs/dendr_avg_len=1000.png', 'png')
    linkage_cent = train_hierarchical(X, Method.CENTROID)
    with open('Linkages/linkage_cent_len=' + str(data_len) +'.pkl', 'wb') as file:
        pickle.dump(linkage_cent, file)
    plots.dendogram(linkage_min, 'Graphs/dendr_cent_len=1000.png', 'png')


if __name__ == "__main__":
    main()
