import copy
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

    #Particionamos las variables continuas usando cuartiles
    qdur = 4
    duration_quants = pd.qcut(df["Duration of Credit (month)"], q=qdur, duplicates='drop')
    duration_cats = range(qdur)
    df["Duration of Credit (month)"] = duration_quants.map(lambda x: duration_cats[duration_quants.cat.categories.get_loc(x)])

    credit_quants = pd.qcut(df["Credit Amount"], q=qdur, duplicates='drop')
    credit_cats = range(qdur)
    df["Credit Amount"] = credit_quants.map(lambda x: credit_cats[credit_quants.cat.categories.get_loc(x)])

    age_quants = pd.qcut(df["Age (years)"], q=qdur, duplicates='drop')
    age_cats = range(qdur)
    df["Age (years)"] = age_quants.map(lambda x: age_cats[age_quants.cat.categories.get_loc(x)])

    data = df.to_numpy()
    np.random.shuffle(data)
    tags = df.columns.to_list()

    randomForest(data, tags)


# Ver en que formato recibe la data de la variable y el dataset
# Se espera que devuelva una fila con los gains
def entropy_gain(variable, data):
    total_len = len(data)
    atribute_set = list(set(variable))
    atribute_indexes = list(map(lambda x: np.where(variable == x), atribute_set))
    classifications = [data] + list(map(lambda x: data[x], atribute_indexes))

    lens = np.array(list(map(lambda x: len(x), classifications)))
    counts = np.array(list(map(lambda x: np.sum(x), classifications)))
    probs = counts / lens

    entropies = probs * np.ma.log2(probs).filled(0) + (1 - probs) * np.ma.log2(
        1 - probs
    ).filled(0)
    entropies[0] = -entropies[0]
    return np.sum(entropies * counts / total_len)


# Se parte de que la variable con la que se clasifica esta en la primer posicion del arreglo
class ID3:
    def __init__(self, data, tags, gain):
        self.data = data
        self.gain = gain
        self.root = self.TreeNode(None, None, 0, data, copy.deepcopy(tags))
        self.generate_tree()

    def generate_tree(self):
        nodes = queue.Queue()
        nodes.put(self.root)

        while not nodes.empty():
            current_node = nodes.get()
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
                    copy.deepcopy(current_node.remaining_tags),
                )

                if not np.all(set[:, 0] == set[0][0]):
                    nodes.put(child)

                current_node.add_child(child)

    def classify(self, point, point_tags, max_depth=None):
        current_node = self.root

        while True:
            if max_depth is not None and current_node.depth == max_depth:
                return current_node.get_classification()

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

            self.tag = self.remaining_tags.pop(max_gain_index)
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
    # NOTE: Uncomment to balance data
    accepted = data[data[:, 0] == 1]
    rejected = data[data[:, 0] == 0]
    data = np.concatenate((accepted, rejected))

    data_len = len(data)
    perc = 0.8

    # Cantidad de arboles del forest
    N = 20

    MAX_DEPTH = 10
    precisions = []

    for n in range(N):
        # Mezclo los datos
        sample = rng.choice(data, replace=True, size=data_len)

        # Separo en train y test
        train = sample[: int(perc * data_len)]
        test = sample[int(perc * data_len) :]

        precision_depth = []

        tree = ID3(train, tags, entropy_gain)

        # Iterar sobre diferentes profundidades del árbol
        for depth in range(1, MAX_DEPTH + 1):
            # Predecir usando el árbol con la profundidad actual
            actual = [tree.classify(x, tags, depth) for x in test]
            expected = [x[0] for x in test]

            matrix = confusion_matrix(expected, actual)
            precision = getPrecision(matrix[0][0], matrix[0][1])

            precision_depth.append((depth, precision))

        precisions.append(precision_depth)

    plot_tree_vs_depth_random_forest(precisions)

    plot_average_precision(precisions)

    sample = rng.choice(data, replace=True, size=data_len)
    train = sample[: int(perc * data_len)]
    test = sample[int(perc * data_len) :]
    tree = ID3(train, tags, entropy_gain)

    for depth in range(1, MAX_DEPTH + 1):
        actual = [tree.classify(x, tags, depth) for x in test]
        expected = [x[0] for x in test]

        matrix = confusion_matrix(expected, actual)

        plot_confusion_matrix(matrix, depth)


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
    plt.title(f"Matriz de confusión profundidad {depth}")
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

    plt.savefig(f"Out/matrix_tree_depth_{depth}.png")


def getPrecision(TP, FP):
    a = TP
    b = TP + FP
    if a == 0:
        return 0
    return a / b


def plot_average_precision(precisions):
    plt.figure()

    # Calcular el promedio y la desviación estándar de las precisiones para cada profundidad
    avg_precisions = {}
    for precision_depth_list in precisions:
        for depth, precision in precision_depth_list:
            if depth not in avg_precisions:
                avg_precisions[depth] = []
            avg_precisions[depth].append(precision)

    avg_depths = []
    avg_values = []
    std_values = []
    for depth, precision_list in avg_precisions.items():
        avg_depths.append(depth)
        avg_values.append(np.mean(precision_list))
        std_values.append(np.std(precision_list))
    
    # Plotear la línea del promedio de las precisiones con barras de error
    plt.errorbar(avg_depths, avg_values, yerr=std_values, fmt="-o")

    plt.xlabel("Depth")
    plt.ylabel("Precision")
    # plt.ylim(0,1)
    plt.title("Average Precision vs. Depth")
    plt.savefig(f"Out/avg_precision_random_forest.png")


def plot_tree_vs_depth_individual(precision_depth, n):
    plt.figure()
    depths, precisions_values = zip(*precision_depth)
    plt.plot(depths, precisions_values, label=f"Tree {n+1}")

    plt.xlabel("Depth")
    plt.ylabel("Precision")
    plt.title(f"Precision vs. Depth - Tree {n+1}")
    plt.savefig(f"Out/precision_tree_{n+1}.png")


def plot_tree_vs_depth_random_forest(precisions):
    plt.figure()
    for i, precision_depth in enumerate(precisions):
        depths, precisions = zip(*precision_depth)
        plt.plot(depths, precisions, label=f"Tree {i+1}")
        plt.plot(depths, precisions)

    plt.xlabel("Depth")
    plt.ylabel("Precision")
    plt.title("Precision vs. Depth")
    plt.legend()
    plt.savefig(f"Out/precision_random_forest.png")

def plot_general_gain(data):
    gains = np.array(list(map(lambda x: entropy_gain(x, data.T[0]), data.T[1:])))

    plt.bar(range(len(gains)), height = gains.round(decimals=3))
    plt.xticks(range(len(gains)))
    plt.ylabel('Gains')
    plt.xlabel('Index')
    plt.title('General Gains')
    plt.savefig(f"Out/general_gain.png")


# FALTA - Ver como particionar variables: duration of credit, credit amount y age
# CHECKEAR - Implementacion de ID3
# CHECKEAR - Implementacion de gain de shannon
# FALTA - Implementacion de Random Forest
# FALTA - Implementacion de matrices

if __name__ == "__main__":
    main()
