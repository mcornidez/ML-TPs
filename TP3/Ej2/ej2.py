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

    old_dataset = np.concatenate((sky, grass, cow), axis=0)
    old_labels = np.concatenate((0*np.ones(len(sky)), 1*np.ones(len(grass)), 2*np.ones(len(cow))))

    #Etiquetas para cada clase
    sky_labels = np.zeros((sky.shape[0], 1))  # 0 para cielo
    grass_labels = np.ones((grass.shape[0], 1))  # 1 para pasto
    cow_labels = np.full((cow.shape[0], 1), 2)  # 2 para vaca

    # Combinar los datos y las etiquetas
    data = np.vstack((sky, grass, cow))
    labels = np.vstack((sky_labels, grass_labels, cow_labels)).flatten()

    # Mezclar los datos y las etiquetas juntos
    dataset = np.hstack((data, labels.reshape(-1, 1)))
    np.random.shuffle(dataset)

    #Separo
    data = dataset[:, :-1] 
    labels = dataset[:, -1].astype(int) 

    perc = 0.8
    train_size = int(perc * len(data))

    dataset_train = data[:train_size]
    labels_train = labels[:train_size]

    dataset_test = data[train_size:]

    #Non-shuffled
    print("Non shuffled")
    clf = svm.SVC()
    clf.fit(dataset[:int(perc*len(old_dataset))], old_labels[:int(perc*len(old_labels))])
    print("Predicting...")
    predictions = clf.predict(old_dataset[-int((1-perc)*len(old_dataset)):])

    print("Predictions old: ", predictions)

    #Shuffled
    print("Shuffled")
    clf = svm.SVC(kernel="linear", C=100)
    clf.fit(data[:int(perc*len(data))], labels[:int(perc*len(labels))])
    print("Predicting...")
    predictions = clf.predict(data[-int((1-perc)*len(data)):])

    print("Predictions new: ", predictions)


if __name__ == "__main__":
    main()
