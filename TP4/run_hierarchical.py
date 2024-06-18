from hierarchical import Method, train_hierarchical
import numpy as np
import utils
import plots
import sys
import pickle
sys.setrecursionlimit(10000)


from scipy.cluster.hierarchy import linkage


def main():
    #data, df = utils.get_data()
    data, df = utils.get_subset()
    df.reset_index(drop=True, inplace=True)
    unique_genres = df['genres'].unique()
    print(unique_genres)
    genre_ids = df['genres'].map(lambda x: np.where(unique_genres == x)[0][0]).to_numpy()


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
    #plots.dendogram(linkage_min, "")
    # Run with library
    #linkage(data, "single")
    #plots.dendogram(linkage(data, "single"), "")
    linkage_min = train_hierarchical(data, Method.MIN, genre_ids)
    linkage_max = train_hierarchical(data, Method.MAX, genre_ids)
    linkage_avg = train_hierarchical(data, Method.AVERAGE, genre_ids)
    linkage_cent = train_hierarchical(data, Method.CENTROID, genre_ids)

    #with open('Linkages/linkage_min_sub_len=' + str(utils.data_len) +'.pkl', 'wb') as file:
    #    pickle.dump(linkage_min, file)
    #with open('Linkages/linkage_max_sub_len=' + str(utils.data_len) +'.pkl', 'wb') as file:
    #    pickle.dump(linkage_max, file)
    #with open('Linkages/linkage_avg_sub_len=' + str(utils.data_len) +'.pkl', 'wb') as file:
    #    pickle.dump(linkage_avg, file)
    #with open('Linkages/linkage_cent_sub_len=' + str(utils.data_len) +'.pkl', 'wb') as file:
    #    pickle.dump(linkage_cent, file)
    
    #plots.dendogram(linkage_min, 'Graphs/Dendr/dendr_min_len=' + str(data_len) + '.png', 'png')
    #plots.dendogram(linkage_max, 'Graphs/Dendr/dendr_max_len=' + str(data_len) + '.png', 'png')
    #plots.dendogram(linkage_avg, 'Graphs/Dendr/dendr_avg_len=' + str(data_len) + '.png', 'png')
    #plots.dendogram(linkage_cent, 'Graphs/Dendr/dendr_cent_len=' + str(data_len) + '.png', 'png')


if __name__ == "__main__":
    main()
