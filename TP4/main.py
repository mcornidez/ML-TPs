import pandas as pd
from datetime import datetime
import numpy as np
from k_means import k_means
import matplotlib.pyplot as plt
#from get_bert_embedding import get_sentence_embedding

subset = ["Action", "Comedy", "Drama"]

# Preguntas!!!!
# Como actualizamos el centroide si no le corresponde ningun elemento (div por 0!!!)
# La varianza nos da valores muy grandes, del orden de 10^20
# Tema de los embeddings de bert
# Como hacemos analisis del error ahora con las variaciones?

#Ver que hacemos con las columnas que son strings:
# No las tenemos en cuenta
# Las tenemos en cuenta y usamos distancia == 0 si hay igualdad y 1 si no
# Usamos embeddings de bert

#Todas las columnas son numericas menos:
#genres, imdb_id, origina_title, overview que son object pero deberian ser strings
#release_date que es object y deberia ser date

#Usar aprox max_epochs = 500 * n siendo n la dimension de los vectores de entrada

numeric_cols = ['budget', 'popularity', 'production_companies', 'production_countries', 'days', 'revenue', 'runtime', 'spoken_languages', 'vote_average', 'vote_count']

def main():
    df = pd.read_csv("movie_data.csv", delimiter=';')

    #Completo campos que tienen na para poder medir distancias
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            mean = df[column].mean()
            std = df[column].std()
            df[column] = (df[column].fillna(mean) - mean)/std
        elif column == "release_date":
            df[column] = pd.to_datetime(df[column])
            #Transformo cada fecha en una cantidad de dias antes o despues del 01/01/2000
            df['days'] = list(map((lambda x: (x - datetime(2000, 1, 1)).days if pd.notna(x) else np.nan), df[column]))
            df['days'] = df['days'].fillna(int(df['days'].mode()[0]))
            mean = df['days'].mean()
            std = df['days'].std()
            df['days'] = (df['days'] - mean)/std
        else:
            df[column] = df[column].astype(str)
            df[column] = df[column].fillna("")

    #Punto 2
    points = df[numeric_cols].to_numpy()
    print_k_means(points)

    #Punto 3
    subset_df = df[df['genres'].isin(subset)]

def print_k_means(points):
    vars = []
    for k in range(1, 11):
        variation, classes, centroids, intermediate_vars = k_means(k, points, False)
        vars.append(variation)
    
    print(vars)

    plt.plot(list(range(1, 11)), vars)
    plt.title('Variation over Parition Number')
    plt.xlabel('Partition Number')
    plt.ylabel('Variation')
    plt.show()
    plt.savefig('Graphs/var_over_k.png')



if __name__ == "__main__":
    main()