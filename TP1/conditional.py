import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Conditional:

    def __init__(self, unknown, known):
        self.unknown = unknown
        self.known = known
        self.learn()

    def learn(self):
        classes = np.unique(known, axis=0)
        #Arreglar esta linea para que known pueda ser multidim...
        class_indexes = list(map(lambda x: np.where(known == x)[0], classes))
        self.counts = np.array(list(map(lambda x: x.size, class_indexes)))
        self.indiv_counts = np.array(list(map(lambda x: unknown[x].transpose().sum(axis=1), class_indexes)))

        #print(classes)
        #print(class_indexes)
        #print(self.counts)
        #print(self.indiv_counts)

