import pandas as pd
import numpy as np
import queue


def main():
    df = pd.read_csv("../Data/german_credit.csv", sep=";")
    data = df.to_numpy()
    print(data[:10])


if __name__ == "__main__":
    main()

#Ver en que formato recibe la data de la variable y el dataset
#Se espera que devuelva una fila con los gains
def entropy_gain(variable, data):
    pass
    

#Se parte de que la variable con la que se clasifica esta en la primer posicion del arreglo
class ID3:
    def __init__(self, data, gain):
        self.data = data
        self.gain = gain
        self.root = self.TreeNode(None, 0, self.data)
        self.generate_tree()

    def generate_tree(self):
        nodes = queue.Queue()
        nodes.put(self.root)

        while not nodes.empty():
            current_node = nodes.get()
            if(len(current_node.data[0]) == 1):
                continue

            subsets = current_node.partition_by_gain(current_node.data)
            for id, set in subsets.items():
                if(len(set) == 0):
                    continue

                child = self.TreeNode(current_node, id, current_node.depth+1, set)

                #Mirar si set[0] es fila o columna
                if(not np.all(set[:, 0] == set[0][0])):
                    nodes.put(child)

                current_node.add_child(child)

    class TreeNode:
        def __init__(self, father, subset_id, depth, data):
            self.father = father
            self.subset_id = subset_id 
            self.depth = depth
            self.data = data 
            self.children = []
            self.classification = None
            self.subsets = None

        def add_child(self, child):
            self.children.append(child)
        
        # Le asigna el valor de la clase que mas aparezca en el arreglo de clases
        def classify(self):
            if(not self.classification is None):
                return self.classification

            unique_elements, counts = np.unique(self.data[0], return_counts=True)
            most_popular_elem = unique_elements[np.argmax(counts)]

            self.classification = most_popular_elem
            return self.classification

        def partition_by_gain(self):
            if not self.subsets is None:
                return self.subsets
            gains = np.array(list(map(lambda x: entropy_gain(x, self.data[0]), self.data.transpose())))
            max_gain_index = np.argmax(gains)
            partition_col = self.data[:, max_gain_index]
            classification_set = list(set(partition_col))
            cropped_data = np.delete(self.data, max_gain_index, index=1)
            partitions_indexes = list(map(lambda x: np.where(partition_col == x), classification_set))
            subsets = list(map(lambda x: cropped_data[x], partitions_indexes))
            self.subsets = {k: v for k, v in zip(classification_set, subsets)}
            return self.subsets


    

#FALTA - Ver como particionar variables: duration of credit, credit amount y age
#FALTA - Implementacion de ID3
#FALTA - Implementacion de gain de shannon
#FALTA - Implementacion de Random Forest
#FALTA - Implementacion de matrices