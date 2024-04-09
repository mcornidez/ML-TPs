from conditionalModel import ConditionalModel 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import time

CATS = ['Nacional', 'Ciencia y Tecnologia', 'Entretenimiento', 'Economia', 'Destacadas', 'Internacional', 'Deportes', 'Salud', 'Noticias destacadas']

def main():
    pre_proc = time.time()
    df = pd.read_excel("Data/cropped_news.xlsx")

    #Por como estan ordenados los datos en este data set tengo una cantidad bien balanceadad de datos de cada categoria
    n = 34660
    perc = 0.80
    threshold = 0

    data = df[["titular", "categoria"]].dropna().to_numpy()
    np.random.shuffle(data)
    titles = data.transpose()[0]
    categorias = list(map(lambda x: CATS.index(x), data.transpose()[1]))

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(titles)
    binary_matrix = np.where(tfidf_matrix.toarray() > threshold, 1, 0)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    train = time.time()
    conditional = ConditionalModel(binary_matrix[:int(n*perc)], categorias[:int(n*perc)])
    test = time.time()
    confusion = np.zeros((len(CATS), len(CATS)))
    res = 0
    roc_confusion = np.zeros((len(CATS), 11, 2, 2))
    for i in range(int(n*perc), int(n)):
        classified = np.array(conditional.naive_classify(binary_matrix[i]))
        # Calculate confusion matrix
        max_index = classified.argmax()
        confusion[categorias[i]][max_index] += 1
        res += (1 if max_index - categorias[i] != 0 else 0)
        # Get data for ROC
        for cat in range(len(CATS)):
            for j in range(0, 11):
                is_really = 0 if cat == categorias[i] else 1
                is_predictically = 0 if classified[cat] > j * 0.1 else 1

                roc_confusion[cat, j, is_really, is_predictically] += 1

    print(confusion)

    # 0 -> TP, 1 -> TN, 2 -> FP, 3 -> FN
    metrics = np.zeros((len(CATS)-1, 4))
    for i in range(len(CATS)-1):
        metrics[i][0] = confusion[i][i]
        metrics[i][2] = confusion.transpose()[i].sum() - metrics[i][0]
        metrics[i][3] = confusion[i].sum() - metrics[i][0]
        metrics[i][1] = confusion.sum().sum() - metrics[i][0] - metrics[i][3] - metrics[i][2]
    metrics = metrics.transpose()


    end = time.time()

    print("Times:")
    print("Preprocessing {}".format(train-pre_proc))
    print("Train {}".format(test-train))
    print("Test {}".format(end-test))
    print(confusion)
    np.savetxt("Out/confusion.csv", confusion, delimiter=',')
    accuracy = (metrics[0] + metrics[1])/(metrics[0] + metrics[1] + metrics[2] + metrics[3])
    precision = metrics[0]/(metrics[0] + metrics[1])
    tpos = metrics[0]/(metrics[0] + metrics[3])
    fpos = metrics[2]/(metrics[2] + metrics[1])
    f1 = (2*precision*tpos)/(precision+tpos)
    metrics = np.array([accuracy, precision, tpos, fpos, f1])
    np.savetxt("Out/metrics.csv", metrics, delimiter=',')
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("TPos Rate: {}".format(tpos))
    print("FPos Rate: {}".format(fpos))
    print("F1: {}".format(f1))

    # Graph ROC
    TVP = roc_confusion[:,:,0,0] / (roc_confusion[:,:,0,0] + roc_confusion[:,:,0,1])
    TFP = roc_confusion[:,:,1,0] / (roc_confusion[:,:,1,0] + roc_confusion[:,:,1,1])

    for i in range(roc_confusion.shape[0]): 
        plt.plot(TFP[i], TVP[i], linestyle='-', marker='o', label=CATS[i])

    plt.plot([0, 1], [0, 1], linestyle='--')
    
    plt.xlabel("Taza de Falsos Positivos")
    plt.ylabel("Taza de Verdaderos Positivos")
    plt.title("Curva ROC")
    plt.legend(loc="best", ncol=2)
    plt.show()

if __name__ == "__main__":
    main()
