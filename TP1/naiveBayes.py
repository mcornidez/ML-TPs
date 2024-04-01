import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Classifier:
    
    def __init__(data, classes, self):
        self.data = data
        self.classes = classes

        for c in classes:

        self.class_prob = ...
        self.individual_prob = ...
        

    def classify(instance):

#Dados los datos almacenados el calculo realizado para clasificar un dato particular es:
#P(C_i|D) = f(i)/Sum_i f(i)
#f(C_i) = (Prod_j individual_prob[D_j, C_i]) * class_prob[C_i]
