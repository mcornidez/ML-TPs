from conditionalModel import ConditionalModel 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import time

CATS = [np.nan, 'Nacional', 'Ciencia y Tecnologia', 'Entretenimiento', 'Economia', 'Destacadas', 'Internacional', 'Deportes', 'Salud', 'Noticias destacadas']

def main():
    start = time.time()
    df = pd.read_excel("Noticias_argentinas.xlsx")

    n = 34668
    perc = 0.90
    threshold = 0.0

    data = df[["titular", "categoria"]].to_numpy()[:n]
    np.random.shuffle(data)
    titles = data.transpose()[0]
    categorias = list(map(lambda x: CATS.index(x), data.transpose()[1]))

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(titles)
    binary_matrix = np.where(tfidf_matrix.toarray() > threshold, 1, 0)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    conditional = ConditionalModel(binary_matrix[:int(n*perc)], categorias[:int(n*perc)])
    res = 0
    for i in range(int(n*perc), int(n)):
        classified = np.array(conditional.naive_classify(binary_matrix[i]))
        #if i < n*perc+20:
        #    print(classified)
        #    print(classified.argmax(axis=0))
        #    print(categorias[i])
        res += (1 if classified.argmax(axis=0) - categorias[i] != 0 else 0)
    print(res/(n*(1-perc)))

    end = time.time()

    print(end - start)

    #for i, file in enumerate(titles):
    #    if i > 20:
    #        break
    #    print(f"Words in file {i+1}:")
    #    feature_index = tfidf_matrix[i,:].nonzero()[1]
    #    tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
    #    for word_index, score in tfidf_scores:
    #        if score > 0.4:
    #            print(f"{feature_names[word_index]}: {score}")







if __name__ == "__main__":
    main()
