from conditionalModel import ConditionalModel 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime


CATS = [np.nan, 'Nacional', 'Ciencia y Tecnologia', 'Entretenimiento', 'Economia', 'Destacadas', 'Internacional', 'Deportes', 'Salud', 'Noticias destacadas']

def main():
    df = pd.read_excel("Noticias_argentinas.xlsx")

    n = 50000
    perc = 0.99
    data = df[["titular", "categoria"]].to_numpy()[:n]
    titles = data.transpose()[0]
    categorias = list(map(lambda x: CATS.index(x), data.transpose()[1]))

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(titles)
    binary_matrix = (tfidf_matrix > 0.3).astype(int).toarray()
    print("First binary")
    #binary_matrix = binary_matrix[:, ~np.all(binary_matrix == 0, axis=0)]
    #print("No waste binary")
    feature_names = tfidf_vectorizer.get_feature_names_out()

    conditional = ConditionalModel(binary_matrix[:int(n*perc)], categorias[:int(n*perc)])
    print("Conditional ready")
    res = 0
    for i in range(int(n*perc), int(n)):
        classified = np.array(conditional.naive_classify(binary_matrix[i]))
        if i < n*perc+20:
            print(classified)
        res += (classified.argmax(axis=0) - categorias[i] != 0)
    print(res)







if __name__ == "__main__":
    main()
