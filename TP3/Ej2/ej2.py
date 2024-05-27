import numpy as np
from PIL import Image
from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

red = np.array(Image.open("Imgs/red.jpg"))
green = np.array(Image.open("Imgs/green.jpg"))
blue = np.array(Image.open("Imgs/blue.jpg"))
COLORS = np.array([red[0, 0], green[0, 0], blue[0, 0]])

def main():
    color_dim = 3
    sky = np.array(Image.open("Imgs/cielo.jpg"))
    sky = sky.reshape((sky.shape[0]*sky.shape[1], color_dim))
    grass = np.array(Image.open("Imgs/pasto.jpg"))
    grass = grass.reshape((grass.shape[0]*grass.shape[1], color_dim))
    cow = np.array(Image.open("Imgs/vaca.jpg"))
    cow = cow.reshape((cow.shape[0]*cow.shape[1], color_dim))

    # Datos no mezclados
    original_dataset = np.concatenate((sky, grass, cow), axis=0)
    original_labels = np.concatenate((0*np.ones(len(sky)), 1*np.ones(len(grass)), 2*np.ones(len(cow))))

    # Mezclar los datos y las etiquetas juntos
    dataset = np.hstack((original_dataset, original_labels.reshape(-1, 1)))
    np.random.shuffle(dataset)

    # Separar los datos y las etiquetas después de mezclar
    data = dataset[:, :-1]
    labels = dataset[:, -1].astype(int)

    perc = 0.8
    train_size = int(perc * len(data))

    dataset_train = data[:train_size]
    labels_train = labels[:train_size]

    dataset_test = data[train_size:]

    # Non-shuffled
    print("Non shuffled")
    clf = svm.SVC()
    clf.fit(original_dataset[:int(perc*len(original_dataset))], original_labels[:int(perc*len(original_labels))])
    print("Predicting...")
    predictions = clf.predict(original_dataset[-int((1-perc)*len(original_dataset)):])

    print("Predictions old: ", predictions)

    # Shuffled
    print("Shuffled")
    clf = svm.SVC(kernel="linear", C=100)
    clf.fit(dataset_train, labels_train)
    print("Predicting...")
    predictions = clf.predict(dataset_test)

    print("Predictions new: ", predictions)

if __name__ == "__main__":
    main()
