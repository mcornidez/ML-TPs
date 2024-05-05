import pandas as pd
import numpy as np
import queue
import math


def main():
    df = pd.read_csv("../Data/german_credit.csv", sep=",")
    #Hay que volver a usar esas columnas una vez que se clasifique bien esa data
    df = df.drop(['Duration of Credit (month)', 'Credit Amount', 'Age (years)'], axis=1)
    data = df.to_numpy()
    np.random.shuffle(data)

    n = len(data)
    perc = 0.8

    train = data[:int(perc*n)]
    test = data[int(perc*n):]
    tags = df.columns.to_list()
    tree = ID3(train, tags, entropy_gain)

    predict = list(map(lambda x: tree.classify(x, tags), test))
    print('porcentaje de aciertos')
    print(1 - sum(abs(predict-test[:, 0]))/len(predict))



#Ver en que formato recibe la data de la variable y el dataset
#Se espera que devuelva una fila con los gains
def entropy_gain(variable, data):
    total_len = len(data)
    atribute_set = list(set(variable))
    atribute_indexes = list(map(lambda x: np.where(variable == x), atribute_set))
    classifications = [data] + list(map(lambda x: data[x], atribute_indexes))
    #classifications = list(map(lambda x: data[x], atribute_indexes))

    lens = np.array(list(map(lambda x: len(x), classifications)))
    counts = np.array(list(map(lambda x: np.sum(x), classifications)))
    probs = counts/lens
    #Hay que usar el logaritmo con alguna base particular?
    entropies = probs * np.nan_to_num(np.log(probs), 0) + (1-probs) * np.nan_to_num(np.log(1-probs), 0)
    entropies[0] = -entropies[0]
    return np.sum(entropies * counts/total_len)


#Se parte de que la variable con la que se clasifica esta en la primer posicion del arreglo
class ID3:
    def __init__(self, data, tags, gain):
        self.data = data
        self.tags = tags
        self.gain = gain
        self.root = self.TreeNode(None, None, 0, data, tags)
        self.generate_tree()

    def generate_tree(self):
        nodes = queue.Queue()
        nodes.put(self.root)

        while not nodes.empty():
            current_node = nodes.get()
            if(len(current_node.data[0]) == 1):
                continue
            
            subsets = current_node.partition_by_gain()
            for id, set in subsets.items():
                if(len(set) == 0):
                    continue

                child = self.TreeNode(current_node, id, current_node.depth+1, set, current_node.remaining_tags)

                #Mirar si set[0] es fila o columna
                if(not np.all(set[:, 0] == set[0][0])):
                    nodes.put(child)

                current_node.add_child(child)
    
    def classify(self, point, point_tags):
        current_node = self.root

        while True:
            if(len(current_node.children) == 0):
                return current_node.get_classification()
            subset_id = point[point_tags.index(current_node.tag)]
            found = False
            for child in current_node.children:
                if child.subset_id == subset_id:
                    current_node = child
                    found = True
            if not found:
                return current_node.father.get_classification()

    class TreeNode:
        def __init__(self, father, subset_id, depth, data, remaining_tags):
            self.father = father
            self.subset_id = subset_id 
            self.depth = depth
            self.data = data 
            self.remaining_tags = remaining_tags
            self.tag = None
            self.children = []
            self.classification = None
            self.subsets = None

        def add_child(self, child):
            self.children.append(child)

        # Le asigna el valor de la clase que mas aparezca en el arreglo de clases
        def get_classification(self):
            if(not self.classification is None):
                return self.classification

            unique_elements, counts = np.unique(self.data[:, 0], return_counts=True)
            most_popular_elem = unique_elements[np.argmax(counts)]

            self.classification = most_popular_elem
            return self.classification

        def partition_by_gain(self):
            if not self.subsets is None:
                return self.subsets
            gains = np.array(list(map(lambda x: entropy_gain(x, self.data[:, 0].transpose()), self.data[:, 1:].transpose())))
            max_gain_index = np.argmax(gains)+1

            partition_col = self.data[:, max_gain_index]
            classification_set = list(set(partition_col))

            self.tag = list(self.remaining_tags).pop(max_gain_index)
            cropped_data = np.delete(self.data, max_gain_index, axis=1)
            partitions_indexes = list(map(lambda x: np.where(partition_col == x), classification_set))
            self.subsets = list(map(lambda x: cropped_data[x], partitions_indexes))
            return {k: v for k, v in zip(classification_set, self.subsets)}


    

#FALTA - Ver como particionar variables: duration of credit, credit amount y age
#CHECKEAR - Implementacion de ID3
#CHECKEAR - Implementacion de gain de shannon
#FALTA - Implementacion de Random Forest
#FALTA - Implementacion de matrices

if __name__ == "__main__":
    main()