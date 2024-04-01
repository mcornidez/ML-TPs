import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Classifier:
    
    def __init__(self, data, classes):
        self.data = data
        self.classes = classes

        class_set = set(classes)
        class_indexes = list(map(lambda x: np.where(classes == x)[0], class_set))
        counts = np.array(list(map(lambda x: x.size, class_indexes)))
        total_count = counts.sum()
        self.class_probs = counts / total_count
        dissected_data = np.array(list(map(lambda x: data[x].transpose().sum(axis=1), class_indexes)))
        self.indiv_prob = dissected_data/counts.reshape((2,1))

        

    def classify(self, data_point):
        complement = 1 - np.array([data_point])
        probs = np.absolute((complement - self.indiv_prob ).prod(axis=1)) * self.class_probs
        return list(map(lambda x: x/probs.sum(), probs))
