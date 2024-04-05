import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ConditionalModel:
    
    def __init__(self, data, classes):
        self.data = data
        self.classes = classes
        self.learn()

    def learn(self):
        class_set = np.unique(self.classes, axis=0)
        class_indexes = list(map(lambda x: np.where(self.classes == x)[0], class_set))

        self.counts = np.array(list(map(lambda x: x.size, class_indexes)))
        self.total_count = self.counts.sum()
        self.class_probs = self.counts / self.total_count

        self.indiv_counts = np.array(list(map(lambda x: self.data[x].transpose().sum(axis=1), class_indexes)))

    def laplace_correction(self, total_count, counts, data_point):

        total_count = total_count.reshape((2,1)) * np.ones(counts.shape)

        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                if self.needs_correction(counts[i][j], total_count[i][j], data_point[j]):
                    counts[i][j] += 1
                    total_count[i][j] += counts.shape[0]
        return total_count, counts

    def needs_correction(self, total_count, counts, data_point):
        if data_point == 0:
            return counts - total_count == 0
        else:
            return counts == 0

    def naive_classify(self, data_point):
        counts, indiv_counts = self.laplace_correction(self.counts, self.indiv_counts, data_point)
        indiv_prob = indiv_counts/counts
        complement = 1 - np.array([data_point])
        probs = np.absolute((complement - indiv_prob ).prod(axis=1)) * self.class_probs
        return list(map(lambda x: x/probs.sum(), probs))
