import numpy as np
from PIL import Image
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
from Ej1.svm import SVM
EPOCHS = 5000
LEARNING_RATE = 0.01

# Cargar imágenes y colores
red = np.array(Image.open("Imgs/red.jpg"))
green = np.array(Image.open("Imgs/green.jpg"))
blue = np.array(Image.open("Imgs/blue.jpg"))
COLORS = np.array([red[0, 0], green[0, 0], blue[0, 0]])

def calculate_metrics(cm):
    # Extraer verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (FP + FN + TP)
    
    # Calcular métricas
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    false_positive_rate = FP / (FP + TN)
    
    # Manejar casos donde la métrica es indefinida
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1 = np.nan_to_num(f1)
    false_positive_rate = np.nan_to_num(false_positive_rate)
    
    # Devolver las métricas promedio (para el caso multiclase)
    return accuracy.mean(), precision.mean(), recall.mean(), f1.mean(), false_positive_rate.mean()

def plot_confusion_matrix(cm, kernel, C, degree=None, normalized=False):
    accuracy, precision, recall, f1, false_positive_rate = calculate_metrics(cm)
    plt.figure(figsize=(10, 6))
    if (normalized):
        filename = f'Out/normalized_{kernel}_{C}.png'
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='.5f')
    else:
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='.6g')
        filename = f'Out/{kernel}_{C}.png'
    if (degree is not None):
        plt.title(f'Kernel: {kernel}, Degree: {degree}, C: {C}\n'
              f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, '
              f'F1-score: {f1:.2f}, FPR: {false_positive_rate:.2f}')
        if (normalized):
            filename = f'Out/normalized_{kernel}_d{degree}_{C}.png'
        else:
            filename = f'Out/{kernel}_d{degree}_{C}.png'
    else:
        plt.title(f'Kernel: {kernel}, C: {C}\n'
              f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, '
              f'F1-score: {f1:.2f}, FPR: {false_positive_rate:.2f}')
    plt.xticks(ticks=np.arange(3)+0.5, labels=['Cielo', 'Pasto', 'Vaca'])
    plt.yticks(ticks=np.arange(3)+0.5, labels=['Cielo', 'Pasto', 'Vaca'], rotation=0)
    
    plt.savefig(filename)
    plt.close()

def plot_points(data, name=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    df_sky = data[data[:, -1] == 0]
    df_grass = data[data[:, -1] == 1]
    df_cow = data[data[:, -1] == 2]
    
    ax.scatter(df_sky[:, 0], df_sky[:, 1], df_sky[:, 2], c='b', label='Cielo')
    ax.scatter(df_grass[:, 0], df_grass[:, 1], df_grass[:, 2], c='g', label='Pasto')
    ax.scatter(df_cow[:, 0], df_cow[:, 1], df_cow[:, 2], c='r', label='Vaca')
    
    ax.set_xlabel('Rojo')
    ax.set_ylabel('Verde')
    ax.set_zlabel('Azul')
    
    plt.legend(loc='upper left')
    plt.grid(True)
    
    if name is not None:
        plt.savefig(f"./Out/scatter_{name}.png")
    
    plt.close()

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
    binary_labels = np.concatenate((0*np.ones(len(sky)), 0*np.ones(len(grass)), 1*np.ones(len(cow))))
    
    labels = binary_labels 
    plotter = plot_binary_confusion_matrix
    use_lib = False
    #labels = original_labels 
    #plotter = plot_confusion_matrix
    
    # Ploteo de puntos antes de mezclar
    data_with_labels = np.hstack((original_dataset, original_labels.reshape(-1, 1)))
    plot_points(data_with_labels, name="TP3_2_a")

    # Mezclar los datos y las etiquetas juntos
    dataset = np.hstack((original_dataset, labels.reshape(-1, 1)))
    np.random.shuffle(dataset)

    # Separar los datos y las etiquetas después de mezclar
    data = dataset[:, :-1]
    labels = dataset[:, -1].astype(int)

    perc = 0.8

    dataset_train = data[:int(perc*len(data))]
    labels_train = labels[:int(perc*len(labels))]

    dataset_test = data[-int((1-perc)*len(data)):]
    labels_test = labels[-int((1-perc)*len(labels)):]

    kernels = ['poly', 'rbf', 'linear']  # TODO: Ver por qué "linear" tarda tanto
    C_values = [0.001, 0.1, 1, 10, 100, 1000]
    poly_degrees = [3, 4, 5]
    pic = np.array(Image.open("Imgs/cow.jpg"))
    flat_pic = pic.reshape((pic.shape[0]*pic.shape[1], color_dim))
    dataset_test = flat_pic
    #painting = clf.predict(flat_pic)
    #painting = np.array(list(map(lambda x: COLORS[int(x)], painting))).reshape((pic.shape[0], pic.shape[1], color_dim))
    #img = Image.fromarray(painting, 'RGB')
    #img.save('Imgs/horse_painting.jpg')

    if not use_lib:
        C = 1
        clf = SVM(dataset_train, labels_train, EPOCHS, LEARNING_RATE, C)
        (weights, best_cost, errors) = clf.train()
        corrected_test = np.append(dataset_test, np.ones((len(dataset_test), 1)), axis = 1).transpose()
        painting = np.sign(np.array(weights[best_cost[1]]) @ corrected_test)
        #clf = svm.SVC()
        #clf.fit(dataset_train, labels_train)
        #painting = clf.predict(dataset_test)
        painting = np.array(list(map(lambda x: COLORS[int(x)], painting))).reshape((pic.shape[0], pic.shape[1], color_dim))
        img = Image.fromarray(painting, 'RGB')
        img.save('Imgs/cow_binary_painting.jpg')
        #cm = confusion_matrix(labels_test, predictions)
        #plotter(cm, "Loss svm", C)
        return



    for kernel in kernels:
        start = time.time()
        for C in C_values:
            if (kernel == 'poly'):
                for d in poly_degrees:
                    clf = svm.SVC(kernel=kernel, C=C, degree=d)
                    clf.fit(dataset_train, labels_train)
                    predictions = clf.predict(dataset_test)
                    cm = confusion_matrix(labels_test, predictions)
                    plot_confusion_matrix(cm, kernel, C, d)
                    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    plot_confusion_matrix(cm_normalized, kernel, C, d, normalized=True)
            else:
                clf = svm.SVC(kernel=kernel, C=C)
                if (kernel == 'linear'):
                    clf = svm.LinearSVC(C=C)  
                clf.fit(dataset_train, labels_train)
                predictions = clf.predict(dataset_test)
                cm = confusion_matrix(labels_test, predictions)
                plot_confusion_matrix(cm, kernel, C)
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                plot_confusion_matrix(cm_normalized, kernel, C, normalized=True)

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
