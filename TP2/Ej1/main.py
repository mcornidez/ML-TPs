import pandas as pd
import numpy as np
import queue
import matplotlib.pyplot as plt

import os

os.makedirs("./Out", exist_ok=True)

# si devolvió el crédito (1) o no (0)
categories = [1, 0]


def main():
    df = pd.read_csv("../Data/german_credit.csv", sep=",")
    # Hay que volver a usar esas columnas una vez que se clasifique bien esa data

    """ 
    Duration of credit:
        0 - < 12 meses
        1 - de 12 a 36 meses
        2 - de 36 a 60 meses
        3 - > 60 meses

    Credit amount:
        0 - < 5000
        1 - de 5000 a 10000
        2 - de 10000 a 15000
        3 - > 15000

    Age:
        0 - < 35 años
        1 - de 35 a 45 años
        2 - de 45 a 55 años
        3 - > 55 años
    """

    df["Duration of Credit (month)"] = df["Duration of Credit (month)"].transform(
        lambda x: 0 if x < 12 else (1 if x <= 36 else (2 if x <= 60 else 3))
    )
    df["Credit Amount"] = df["Credit Amount"].transform(
        lambda x: 0 if x < 5000 else (1 if x <= 10000 else (2 if x <= 15000 else 3))
    )
    df["Age (years)"] = df["Age (years)"].transform(
        lambda x: 0 if x < 35 else (1 if x <= 45 else (2 if x <= 55 else 3))
    )

    # df = df.drop(['Duration of Credit (month)', 'Credit Amount', 'Age (years)'], axis=1)
    data = df.to_numpy()
    np.random.shuffle(data)

    n = len(data)
    perc = 0.8

    train = data[: int(perc * n)]
    test = data[int(perc * n) :]
    tags = df.columns.to_list()
    tree = ID3(train, tags, entropy_gain)

    # predict = list(map(lambda x: tree.classify(x, tags), test))
    # print('porcentaje de aciertos')
    # print(1 - sum(abs(predict-test[:, 0]))/len(predict))

    randomForest(data, tags)


# Ver en que formato recibe la data de la variable y el dataset
# Se espera que devuelva una fila con los gains
def entropy_gain(variable, data):
    total_len = len(data)
    atribute_set = list(set(variable))
    atribute_indexes = list(map(lambda x: np.where(variable == x), atribute_set))
    classifications = [data] + list(map(lambda x: data[x], atribute_indexes))
    # classifications = list(map(lambda x: data[x], atribute_indexes))

    lens = np.array(list(map(lambda x: len(x), classifications)))
    counts = np.array(list(map(lambda x: np.sum(x), classifications)))
    probs = counts / lens
    # Hay que usar el logaritmo con alguna base particular?
    entropies = probs * np.nan_to_num(np.log2(probs)) + (1 - probs) * np.nan_to_num(
        np.log2(1 - probs)
    )
    entropies[0] = -entropies[0]
    return np.sum(entropies * counts / total_len)


# Se parte de que la variable con la que se clasifica esta en la primer posicion del arreglo
class ID3:
    def __init__(self, data, tags, gain, max_depth=None):
        self.data = data
        self.tags = tags
        self.gain = gain
        self.root = self.TreeNode(None, None, 0, data, tags)
        self.max_depth = max_depth
        self.generate_tree()

    def generate_tree(self):
        nodes = queue.Queue()
        nodes.put(self.root)

        while not nodes.empty():
            current_node = nodes.get()
            if self.max_depth is not None and current_node.depth >= self.max_depth:
                continue
            if len(current_node.data[0]) == 1:
                continue

            subsets = current_node.partition_by_gain()
            for id, set in subsets.items():
                if len(set) == 0:
                    continue

                child = self.TreeNode(
                    current_node,
                    id,
                    current_node.depth + 1,
                    set,
                    current_node.remaining_tags,
                )

                # Mirar si set[0] es fila o columna
                if not np.all(set[:, 0] == set[0][0]):
                    nodes.put(child)

                current_node.add_child(child)

    def classify(self, point, point_tags):
        current_node = self.root

        while True:
            if len(current_node.children) == 0:
                return current_node.get_classification()
            subset_id = point[point_tags.index(current_node.tag)]
            found = False
            for child in current_node.children:
                if child.subset_id == subset_id:
                    current_node = child
                    found = True
            if not found:
                return current_node.father.get_classification()  # type: ignore

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
            if not self.classification is None:
                return self.classification

            unique_elements, counts = np.unique(self.data[:, 0], return_counts=True)
            most_popular_elem = unique_elements[np.argmax(counts)]

            self.classification = most_popular_elem
            return self.classification

        def partition_by_gain(self):
            if not self.subsets is None:
                return self.subsets
            gains = np.array(
                list(
                    map(
                        lambda x: entropy_gain(x, self.data[:, 0].transpose()),
                        self.data[:, 1:].transpose(),
                    )
                )
            )
            max_gain_index = np.argmax(gains) + 1

            partition_col = self.data[:, max_gain_index]
            classification_set = list(set(partition_col))

            self.tag = list(self.remaining_tags).pop(max_gain_index)
            cropped_data = np.delete(self.data, max_gain_index, axis=1)
            partitions_indexes = list(
                map(lambda x: np.where(partition_col == x), classification_set)
            )
            self.subsets = list(map(lambda x: cropped_data[x], partitions_indexes))
            return {k: v for k, v in zip(classification_set, self.subsets)}


def getPrecision(tree, data, tags):  # type: ignore
    predictions = [tree.classify(x, tags) for x in data]
    true_labels = [x[0] for x in data]
    accuracy = sum(
        1 for i in range(len(data)) if true_labels[i] == predictions[i]
    ) / len(data)
    return accuracy


rng = np.random.default_rng()


def randomForest(data, tags):
    data_len = len(data)
    perc = 0.8

    # Cantidad de arboles del forest
    N = 10

    MAX_DEPTH = 10

    for n in range(N):
        # Mezclo los datos
        sample = rng.choice(data, replace=True, size=data_len)

        # Separo en train y test
        train = sample[: int(perc * data_len)]
        test = sample[int(perc * data_len) :]

        precision_depth = []

        # Iterar sobre diferentes profundidades del árbol
        for depth in range(1, MAX_DEPTH + 1):
            # Construir el árbol con la profundidad actual
            tree = ID3(train, tags, entropy_gain, max_depth=depth)

            # Predecir usando el árbol
            actual = [tree.classify(x, tags) for x in test]
            expected = [x[0] for x in test]

            matrix = confusion_matrix(expected, actual)
            precision = getPrecision(matrix[0][0], matrix[0][1])

            precision_depth.append((depth, precision))

            # Plot de la matriz de confusión
            plot_confusion_matrix(matrix, depth)

        # Graficar la precisión vs profundidad para este árbol
        plt.figure()
        depths, precisions = zip(*precision_depth)
        plt.plot(depths, precisions, label=f"Tree {n+1}")

        plt.xlabel("Depth")
        plt.ylabel("Precision")
        plt.title(f"Precision vs. Depth - Tree {n+1}")
        plt.legend()
        plt.savefig(f"Out/precision_{n+1}.png")


def confusion_matrix(expected, actual):
    # 0 -> TP, 1 -> TN, 2 -> FP, 3 -> FN
    matrix = np.zeros((2, 2))
    # TP
    matrix[0][0] = sum(
        (expected[i] == 1) and (actual[i] == 1) for i in range(len(expected))
    )
    # TN
    matrix[1][1] = sum(
        (expected[i] == 0) and (actual[i] == 0) for i in range(len(expected))
    )
    # FP
    matrix[0][1] = sum(
        (expected[i] == 0) and (actual[i] == 1) for i in range(len(expected))
    )
    # FN
    matrix[1][0] = sum(
        (expected[i] == 1) and (actual[i] == 0) for i in range(len(expected))
    )
    return matrix


def plot_confusion_matrix(matrix, depth):
    plt.figure()
    plt.imshow(matrix, cmap="Blues", interpolation="nearest")
    plt.title(f"Matriz de confusión árbol {depth}")
    plt.colorbar()

    plt.xlabel("Predicted")
    plt.ylabel("Real")
    plt.xticks(
        np.arange(matrix.shape[1]),
        list(map(str, range(matrix.shape[1]))),
    )
    plt.yticks(
        np.arange(matrix.shape[0]),
        list(map(str, range(matrix.shape[0]))),
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                str(int(matrix[i, j])),
                ha="center",
                va="center",
                color="black",
            )

    plt.savefig(f"Out/matrix_{depth}.png")


def getPrecision(TP, FP):
    a = TP
    b = TP + FP
    if a == 0:
        return 0
    return a / b


# FALTA - Ver como particionar variables: duration of credit, credit amount y age
# CHECKEAR - Implementacion de ID3
# CHECKEAR - Implementacion de gain de shannon
# FALTA - Implementacion de Random Forest
# FALTA - Implementacion de matrices

if __name__ == "__main__":
    main()
