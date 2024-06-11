import pandas as pd
from datetime import datetime
import numpy as np
from k_means import k_means
#from get_bert_embedding import get_sentence_embedding

subset = ["Action", "Comedy", "Drama"]

#Ver que hacemos con las columnas que son strings:
# No las tenemos en cuenta
# Las tenemos en cuenta y usamos distancia == 0 si hay igualdad y 1 si no
# Usamos embeddings de bert

#Todas las columnas son numericas menos:
#genres, imdb_id, origina_title, overview que son object pero deberian ser strings
#release_date que es object y deberia ser date

numeric_cols = ['budget', 'popularity', 'production_companies', 'production_countries', 'days', 'revenue', 'runtime', 'spoken_languages', 'vote_average', 'vote_count']

def main():
    df = pd.read_csv("movie_data.csv", delimiter=';')

    #Completo campos que tienen na para poder medir distancias
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].mean())
        elif column == "release_date":
            df[column] = pd.to_datetime(df[column])
            #Transformo cada fecha en una cantidad de dias antes o despues del 01/01/2000
            df['days'] = list(map((lambda x: (x - datetime(2000, 1, 1)).days if pd.notna(x) else np.nan), df[column]))
            df['days'] = df['days'].fillna(int(df['days'].mode()[0]))
        else:
            df[column] = df[column].astype(str)
            df[column] = df[column].fillna("")

    #Punto 2
    points = df[numeric_cols].to_numpy()
    variation, classes, centroids = k_means(5, points)

    #Punto 3
    subset_df = df[df['genres'].isin(subset)]



if __name__ == "__main__":
    main()