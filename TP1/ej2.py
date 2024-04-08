from conditionalModel import ConditionalModel 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import time

CATS = [np.nan, 'Nacional', 'Ciencia y Tecnologia', 'Entretenimiento', 'Economia', 'Destacadas', 'Internacional', 'Deportes', 'Salud', 'Noticias destacadas']

def main():
    pre_proc = time.time()
    df = pd.read_excel("Data/cropped_news.xlsx")

    #Por como estan ordenados los datos en este data set tengo una cantidad bien balanceadad de datos de cada categoria
    n = 34668
    perc = 0.80
    threshold = 0

    data = df[["titular", "categoria"]].to_numpy()
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
    for i in range(int(n*perc), int(n)):
        classified = np.array(conditional.naive_classify(binary_matrix[i]))
        max_index = classified.argmax()
        confusion[categorias[i]][max_index] += 1
        res += (1 if max_index - categorias[i] != 0 else 0)

    confusion = confusion[1:].transpose()[1:].transpose()
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

    print("Accuracy percentage {}".format(1 - res/(n*(1-perc))))
    print("Preprocessing {}".format(train-pre_proc))
    print("Train {}".format(test-train))
    print("Test {}".format(end-test))
    print(confusion)
    accuracy = (metrics[0] + metrics[1])/(metrics[0] + metrics[1] + metrics[2] + metrics[3])
    precision = metrics[0]/(metrics[0] + metrics[1])
    tpos = metrics[0]/(metrics[0] + metrics[3])
    fpos = metrics[2]/(metrics[2] + metrics[1])
    f1 = (2*precision*tpos)/(precision+tpos)
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("TPos Rate: {}".format(tpos))
    print("FPos Rate: {}".format(fpos))
    print("F1: {}".format(f1))


if __name__ == "__main__":
    main()
