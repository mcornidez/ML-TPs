from conditionalModel import ConditionalModel 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

CATS = [np.nan, 'Nacional', 'Ciencia y Tecnologia', 'Entretenimiento', 'Economia', 'Destacadas', 'Internacional', 'Deportes', 'Salud', 'Noticias destacadas']

def main():
    df = pd.read_excel("Noticias_argentinas.xlsx")

    data = df[["titular", "categoria"]].to_numpy()[:50000]
    titles = data.transpose()[0]
    categorias = list(map(lambda x: CATS.index(x), data.transpose()[1]))

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(titles)
    binary_matrix = (tfidf_matrix > 0.25).astype(int).toarray()
    feature_names = tfidf_vectorizer.get_feature_names_out()

    conditional = ConditionalModel(binary_matrix[:45000], categorias[:45000])
    print()
    res = 0
    #print(classified[45000, 45008])
    for i in range(45000, 45100):
        classified = np.array(conditional.naive_classify(binary_matrix[i]))
        res += (classified.argmax(axis=0) - categorias[i] != 0)
    print(res)







if __name__ == "__main__":
    main()
