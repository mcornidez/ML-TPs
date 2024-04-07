import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ConditionalModel:
    
    def __init__(self, data, classes):
        self.data = data
        self.classes = classes
        self.learn()

    def learn(self):
        self.class_set = np.unique(self.classes, axis=0)
        self.class_indexes = list(map(lambda x: np.where(list(map(lambda x: np.all(x), np.equal(self.classes, x))))[0], self.class_set))

        self.counts = np.array(list(map(lambda x: x.size, self.class_indexes)))
        self.total_count = self.counts.sum()
        self.class_probs = self.counts / self.total_count

        self.indiv_counts = np.array(list(map(lambda x: self.data[x].transpose().sum(axis=1), self.class_indexes)))

    def laplace_correction(self, total_count, counts, data_point):
        data_point = data_point.reshape((data_point.shape[0],1)).transpose()
        total_count = total_count.reshape((total_count.shape[0],1))

        needs_correction = np.where(counts - total_count * (1 - np.array(data_point)) == 0, 1, 0)
        counts += needs_correction
        total_count = total_count + needs_correction * 2

        return total_count, counts

    def naive_classify(self, data_point):
        counts, indiv_counts = self.laplace_correction(self.counts, self.indiv_counts, data_point)
        indiv_prob = indiv_counts/counts
        complement = 1 - np.array([data_point])
        probs = np.absolute((complement - indiv_prob ).prod(axis=1)) * self.class_probs
        return list(map(lambda x: x/probs.sum(), probs))
    
    def calculate_conditional(self):
        self.class_probs = np.squeeze(self.indiv_counts)/self.counts
        return self.class_set, self.class_probs
