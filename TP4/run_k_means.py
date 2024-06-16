import utils
from k_means import k_means
import numpy as np
import matplotlib.pyplot as plt

def main():
    points, df = utils.get_data()
    #points, df = utils.get_subset()
    #print_k_means(points)
    variation, variations, classes, centroids, intermediate_vars = k_means(7, points, False)
    print(utils.numeric_cols)
    print(variations)
    print(centroids)


def print_k_means(points):
    runs = 5
    vars = np.zeros((runs, 10))
    for i in range(runs):
        for k in range(1, 11):
            variation, variations, classes, centroids, intermediate_vars = k_means(k, points, False)
            vars[i, k-1] = variation
            print(vars)
    
    mean = vars.sum(axis=0)/runs
    print(mean)

    plt.plot(list(range(1, 11)), mean)
    plt.title('Variation over Parition Number')
    plt.xlabel('Partition Number')
    plt.ylabel('Variation')
    plt.show()
    #plt.savefig('Graphs/var_over_k.png')


if __name__ == "__main__":
    main()