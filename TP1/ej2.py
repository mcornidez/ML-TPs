from conditionalModel import ConditionalModel 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import time

CATS = [np.nan, 'Nacional', 'Ciencia y Tecnologia', 'Entretenimiento', 'Economia', 'Destacadas', 'Internacional', 'Deportes', 'Salud', 'Noticias destacadas']

def main():
    pre_proc = time.time()
    df = pd.read_excel("cropped_news.xlsx")

    #Por como estan ordenados los datos en este data set tengo una cantidad bien balanceadad de datos de cada categoria
    n = 34668
    perc = 0.80
    threshold = 0.1

    data = df[["titular", "categoria"]].to_numpy()
    np.random.shuffle(data)
    titles = data.transpose()[0]
    categorias = list(map(lambda x: CATS.index(x), data.transpose()[1]))

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(titles)
    binary_matrix = np.where(tfidf_matrix.toarray() > threshold, 1, 0)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    proc = time.time()
    conditional = ConditionalModel(binary_matrix[:int(n*perc)], categorias[:int(n*perc)])
    res = 0
    for i in range(int(n*perc), int(n)):
        classified = np.array(conditional.naive_classify(binary_matrix[i]))
        res += (1 if classified.argmax(axis=0) - categorias[i] != 0 else 0)

    end = time.time()

    print("Accuracy percentage {}".format(1 - res/(n*(1-perc))))
    print("Preprocessing {}".format(proc-pre_proc))
    print("Processing {}".format(end-proc))


if __name__ == "__main__":
    main()
