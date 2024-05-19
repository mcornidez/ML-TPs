import numpy as np

#Se asum que la probabilidad de que el clasificador arroje 0 es depreciable
def generate_classified(amount, dim, vec_classifier):
    points = np.random.uniform(-1, 1, (amount, dim))
    classified_points = np.concatenate((points, vec_classifier(points[:, dim-1]).reshape((amount, 1))), axis=1) 
    return classified_points

def main():
    dim = 2
    total = 10
    classif = 8
    misclassif = total - classif

    TP3_1 = generate_classified(total, dim, np.sign)
    print("TP3_1")
    print(TP3_1)

    #Los primeros classif elementos estaran bien clasificados y los ultimos misclassif mal clasificados
    classified_set = generate_classified(classif, dim, np.sign)
    misclassified_set = generate_classified(misclassif, dim, lambda x: -np.sign(x))

    TP3_2 = np.concatenate((classified_set, misclassified_set), axis=0)
    print("TP3_2")
    print(TP3_2)

if __name__ == "__main__":
    main()
