import pandas as pd
import numpy as np
from datetime import datetime

subset = ["Action", "Comedy", "Drama"]

numeric_cols = ['budget', 'popularity', 'production_companies', 'production_countries', 'days', 'revenue', 'runtime', 'spoken_languages', 'vote_average', 'vote_count']

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
        df['days'] = df['days'].fillna(int(df['days'].mode()[0])) # type: ignore
        mean = df['days'].mean()
        std = df['days'].std()
        df['days'] = (df['days'] - mean)/std
    else:
        df[column] = df[column].astype(str)
        df[column] = df[column].fillna("")

def get_data(): 
    return df[numeric_cols].to_numpy(), df

subset_df = df[df['genres'].isin(subset)]

def get_subset():
    return subset_df[numeric_cols].to_numpy(), subset_df