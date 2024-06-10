import numpy as np
import random

def k_means(k, points):
    centroids = points[init_centroids(k, points.shape[0])]
    print(centroids.shape)
    classes = []

    while True:
        diffs = list(map(lambda x: diff(x, centroids), points))
        classes = list(map(lambda x: np.where(x == min(x))[0][0], diffs))
        #No numpy version vvv
        #classes = list(map(lambda x: x.index(min(x)), diffs))
        unique, count = np.unique(np.array(classes), return_counts=True)
        indexes = list(map(lambda x: np.where(classes == x), unique))
        new_centroids = np.array(list(map(lambda x: points[indexes[x]].sum(axis=0)/count[x], range(len(indexes)))))
        print("a")
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return classes, centroids




def init_centroids(k, points):
    #No numpy version vvv
    #return random.sample(points, k)
    return np.random.choice(points, size=k, replace=False)

def diff(x, centroids):
    x = np.array(x)
    centroids = np.array(centroids)

    squared_diff = (centroids - x)**2
    return np.sum(squared_diff, axis=1)

def calculate_variation(classes, points):
    pass