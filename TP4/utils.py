import pandas as pd
import numpy as np
from datetime import datetime

subset = ["Action", "Comedy", "Drama"]

numeric_cols = ['budget', 'popularity', 'production_companies', 'production_countries', 'revenue', 'runtime', 'spoken_languages', 'vote_average', 'vote_count']

df = pd.read_csv("movie_data.csv", delimiter=';')

use_genres = False
use_date = True

#Proceso release_date
if use_date:
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['days'] = list(map((lambda x: (x - datetime(2000, 1, 1)).days if pd.notna(x) else np.nan), df['release_date']))
    df['days'] = df['days'].fillna(int(df['days'].mode()[0])) # type: ignore
    numeric_cols.append('days')


#Proceso genres
if use_genres:
    df['genres'] = df['genres'].astype(str)
    df['genres'] = df['genres'].fillna("")
    unique_genres = df['genres'].unique()
    df['genre_id'] = df['genres'].map(lambda x: np.where(unique_genres == x)[0][0])
    numeric_cols.append('genre_id')

for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        mean = df[column].mean()
        std = df[column].std()
        df[column] = (df[column].fillna(mean) - mean)/std

def get_data(): 
    return df[numeric_cols].to_numpy(), df

subset_df = df[df['genres'].isin(subset)]

def get_subset():
    return subset_df[numeric_cols].to_numpy(), subset_df