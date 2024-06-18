import utils
from k_means import k_means
import numpy as np
import matplotlib.pyplot as plt

def main():
    points, df = utils.get_data()
    #points, df = utils.get_subset()
    df.reset_index(drop=True, inplace=True)
    #print_k_means(points)


    variation, variations, classes, centroids, intermediate_vars = k_means(12, points, False)

    #comedy_cls = classes[genres_df.index[genres_df['genres'] == 'Comedy']]
    comedy_cls = classes[df.index[df['genres'] == 'Comedy']]
    unique, counts = np.unique(comedy_cls, return_counts=True)
    print(counts)
    drama_cls = classes[df.index[df['genres'] == 'Drama']]
    unique, counts = np.unique(drama_cls, return_counts=True)
    print(counts)
    action_cls = classes[df.index[df['genres'] == 'Action']]
    unique, counts = np.unique(action_cls, return_counts=True)
    print(counts)

    #print(utils.numeric_cols)
    print(variation)


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