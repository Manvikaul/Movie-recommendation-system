# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:01:50 2020

@author: DELL
"""

import pandas as pd

movies=pd.read_csv(r'C:\Users\DELL\Documents\Movie-recommendation-system\movies.csv')
ratings=pd.read_csv(r'C:\Users\DELL\Documents\Movie-recommendation-system\ratings.csv')
movies.head(50)
movies.columns
ratings.columns

ratings=pd.merge(movies,ratings).drop(['genres','timestamp'],axis=1)
ratings.head()
ratings.columns

user_ratings=ratings.pivot_table(index=['userId'],columns=['title'],values='rating')
user_ratings.head()

#Drop all movies rated by less than 10 users
#Fill the remaining NaNs with 0
user_ratings=user_ratings.dropna(thresh=5,axis=1).fillna(0)
user_ratings.head()

item_similarity_df=user_ratings.corr(method='pearson')
item_similarity_df.head(50)

def get_similar_movies(movie_name,user_rating):
    similar_score=item_similarity_df[movie_name]*(user_rating-2.5)
    similar_score=similar_score.sort_values(ascending=False)
    
    return similar_score


sample_action_lover_user=[('(500) Days of Summer (2009)',2),
                          ('12 Years a Slave (2013)',4),
                          ('2 Fast 2 Furious (Fast and the Furious 2, The) (2003)',5)]

similar_movies=pd.DataFrame()

for movie,rating in sample_action_lover_user:
    similar_movies=similar_movies.append(get_similar_movies(movie,rating),ignore_index=True)
    
similar_movies.head()
similar_movies.sum().sort_values(ascending=False)
    
    