import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Classifier:
    
    def __init__(self, data, classes):
        self.data = data
        self.classes = classes
        self.learn(data, classes)

    def learn(self, data, classes):
        class_set = set(classes)
        class_indexes = list(map(lambda x: np.where(classes == x)[0], class_set))

        counts = np.array(list(map(lambda x: x.size, class_indexes)))
        total_count = counts.sum()
        self.class_probs = counts / total_count

        indiv_counts = np.array(list(map(lambda x: data[x].transpose().sum(axis=1), class_indexes)))
        counts, indiv_counts = self.laplace_correction(counts, indiv_counts)
        print(indiv_counts)
        print(counts)
        self.indiv_prob = indiv_counts/counts.reshape((2,1))

    def laplace_correction(self, total_count, counts):

        complement = total_count.reshape((2,1)) - counts
        complement[complement != 0] = 1

        for i in range(counts.shape[0]-1):
            if self.needs_correction(counts[i], total_count[i]):
                counts[i] += complement[i]
                total_count[i] += 1
        return total_count, counts

    def needs_correction(self, total_count, counts):
        return np.isin(0, (counts - total_count) * counts)

    def classify(self, data_point):
        complement = 1 - np.array([data_point])
        probs = np.absolute((complement - self.indiv_prob ).prod(axis=1)) * self.class_probs
        return list(map(lambda x: x/probs.sum(), probs))
