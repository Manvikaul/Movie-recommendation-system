# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 22:30:56 2020

@author: DELL
"""

import numpy as np
import pandas as pd

df=pd.read_csv(r"C:\Users\DELL\Documents\movie Recommendation system\movie_dataset.csv")
print(df.head())
print(df.columns)

features=['keywords','cast','director','genres']

for feature in features:
    df[feature]=df[feature].fillna('');

def combineFeatures(row):
    try:
        return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
    except:
        print("error",row)

df['combined_features']=df.apply(combineFeatures,axis=1)

print('Combined features: ',df['combined_features'].head())

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
countMatrix=cv.fit_transform(df['combined_features'])
    
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim=cosine_similarity(countMatrix)    

def get_index_from_title(title):
    return df[df.title==title]["index"].values[0]

def get_title_from_index(index):
    return df[df.index==index]['title'].values[0]

print(df['title'].head(10))

movie_user_likes='Tangled'

movie_index=get_index_from_title(movie_user_likes)
print(movie_index)

similar_movies=list(enumerate(cosine_sim[movie_index]))
sorted_similar_movies=sorted(similar_movies,key=lambda x: x[1],reverse=True)

i=0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    i=i+1
    if i>50:
        break