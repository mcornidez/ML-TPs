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

    dataset_train = data[:int(perc*len(data))]
    labels_train = labels[:int(perc*len(labels))]

    dataset_test = data[-int((1-perc)*len(data)):]
    labels_test = labels[-int((1-perc)*len(labels)):]

    kernels = ['poly', 'rbf']  #TODO: Ver por que "linear" tarda tanto
    C_values = [0.1, 1, 10, 100] 

    for kernel in kernels:
        for C in C_values:
            clf = svm.SVC(kernel=kernel, C=C)
            clf.fit(dataset_train, labels_train)
            predictions = clf.predict(dataset_test)
            cm = confusion_matrix(labels_test, predictions)
            plot_confusion_matrix(cm, kernel, C)

def plot_confusion_matrix(cm, kernel, C):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2f')
    plt.title(f'Kernel: {kernel}, C: {C}')
    plt.xticks(ticks=np.arange(3)+0.5, labels=['Cielo', 'Pasto', 'Vaca'])
    plt.yticks(ticks=np.arange(3)+0.5, labels=['Cielo', 'Pasto', 'Vaca'], rotation=0)
    
    filename = f'Out/{kernel}_{C}.png'
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    main()
