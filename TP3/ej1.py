import numpy as np

#Se asum que la probabilidad de que el clasificador arroje 0 es depreciable
def generate_classified(amount, dim, vec_classifier):
    points = np.random.uniform(-1, 1, (amount, dim))
    classified_points = np.concatenate((points, vec_classifier(points[:, dim-1]).reshape((amount, 1))), axis=1) 
    return classified_points

def main():
    dim = 2
    amount = 10
    classified_set = generate_classified(amount, dim, np.sign)
    print(classified_set)

if __name__ == "__main__":
    main()
