from hierarchical import Method, train_hierarchical
import numpy as np
from utils import get_data
import plots

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

    linkage_min = train_hierarchical(X, Method.MIN)
    plots.dendogram(linkage_min, "")
    # Run with library
    plots.dendogram(linkage(data, "single"), "")
    linkage_max = train_hierarchical(data, Method.MAX)
    linkage_avg = train_hierarchical(data, Method.AVERAGE)
    linkage_centroid = train_hierarchical(data, Method.CENTROID)


if __name__ == "__main__":
    main()
