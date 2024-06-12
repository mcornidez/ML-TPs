import numpy as np
import random

def k_means(k, points, accumulate_var):
    centroids = points[init_centroids(k, points.shape[0])]
    classes = []

    intermediate_var = []

    while True:
        diffs = list(map(lambda x: centroid_diff(x, centroids), points))
        classes = np.argmin(diffs, axis=1)
        indexes = list(map(lambda x: np.where(classes == x), range(k)))
        counts = []
        for i in range(k):
            count = np.where(classes == i)[0].shape[0]
            counts.append(count if count != 0 else 1)
        new_centroids = np.array(list(map(lambda x: points[indexes[x]].sum(axis=0)/counts[x], range(len(indexes)))))
        if accumulate_var == True:
            var = sum(calculate_variation(classes, points))
            intermediate_var.append(var)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    variation = sum(calculate_variation(classes, points))

    return variation, classes, centroids, intermediate_var


def init_centroids(k, points):
    centroids = np.random.choice(points, size=k, replace=False)
    return centroids

def centroid_diff(x, centroids):
    x = np.array(x)
    centroids = np.array(centroids)

    squared_diff = (centroids - x)**2
    return np.sum(squared_diff, axis=1)

def calculate_variation(classes, points):
    unique = np.unique(np.array(classes))
    indexes = list(map(lambda x: np.where(classes == x), unique))
    vars = list(map(lambda x: single_variation(points[x]), indexes))
    return vars

def single_variation(points):
    points = np.array(points)
    points = np.reshape(points, (points.shape[1], points.shape[0], 1))
    point_count = points.shape[1]
    ext_points = points * np.ones((points.shape[0], points.shape[1], points.shape[1]))
    #diff = np.triu(ext_points - np.moveaxis(ext_points, 0, 1))
    diff_triu = np.triu(ext_points - np.moveaxis(ext_points, 1, 2))
    #diff_tril = np.tril(ext_points - np.moveaxis(ext_points, 1, 2))
    #res = np.sum((diff)**2)/point_count
    res_triu = np.sum(diff_triu**2)/point_count
    #res_tril = np.sum(diff_tril**2)/point_count
    return res_triu
