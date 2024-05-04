import pandas as pd
import numpy as np
import queue


def main():
    df = pd.read_csv("../Data/german_credit.csv", sep=";")
    data = df.to_numpy()
    print(data[:10])



if __name__ == "__main__":
    main()

#Devuelve 
def entropy_gain():
    pass
    

class ID3:

    def __init__(self, data, gain):
        self.data = data
        self.gain = gain
        self.root = self.TreeNode(None, 0, self.data)
        #Se parte de que la variable con la que se clasifica esta en la primer posicion del arreglo
        self.generate_tree()

    def generate_tree(self):
        nodes = queue.Queue()
        nodes.put(self.root)

        while not nodes.empty():
            current_node = nodes.get()
            subsets = self.partition_by_gain(current_node.data)
            #Caso base con una unica clase
            if(len(subsets) > 1):
                for i, set in enumerate(subsets):
                    child = self.TreeNode(current_node, current_node.depth+1, set)

                    if(len(set) == 0):
                        child.classification = current_node.classify()
                        continue

                    if(np.all(set[0] == set[0][0])):
                        child.classification = set[0][0]
                    else:
                        nodes.put(child)

                    current_node.add_child(child)

    # Devuelve un arreglo en el que cada elemento es una subset obtenido en base a la variable con mayor gain
    # Solo devuelve un arreglo de un solo elemento si la unica varaible restante es la de clasificacion
    def partition_by_gain(self, data):
        pass


    class TreeNode:
        def __init__(self, father, depth, data):
            self.father = father
            self.depth = depth
            self.data = data 
            self.children = []
            self.classification = None

        def add_child(self, child):
            self.children.append(child)
        
        def classify(self):
            if(not self.classification is None):
                return self.classification

            unique_elements, counts = np.unique(self.data[0], return_counts=True)
            most_popular_elem = unique_elements[np.argmax(counts)]

            self.classification = most_popular_elem
            return self.classification
    
    def classify(self, data):
        pass



#Ver como particionar variables: duration of credit, credit amount y age
#Implementacion de ID3
#Implementacion de gain de shannon
#Implementacion de Random Forest
#Implementacion de matrices