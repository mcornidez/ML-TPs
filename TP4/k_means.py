import numpy as np
import random

def k_means(k, points, accumulate_var):
    unique = np.unique(points ,axis=0)
    centroids = unique[init_centroids(k, unique.shape[0])]
    classes = []

    intermediate_var = []

    while True:
        diffs = list(map(lambda x: centroid_diff(x, centroids), points))
        classes = np.argmin(diffs, axis=1)
        unique, counts = np.unique(classes, return_counts=True)
        indexes = list(map(lambda x: np.where(classes == x), unique))
        new_centroids = np.array(list(map(lambda x: points[indexes[x]].sum(axis=0)/counts[x], range(len(unique)))))
        if accumulate_var == True:
            var = sum(calculate_variation(classes, points))
            intermediate_var.append(var)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    variations = calculate_variation(classes, points)

    return sum(variations), variations, classes, centroids, intermediate_var


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
    diff_triu = np.triu(ext_points - np.moveaxis(ext_points, 1, 2))
    res_triu = np.sum(diff_triu**2)/point_count
    return res_triu
