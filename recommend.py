import pandas as pd
import numpy as np
import json
import re
import sys
import itertools

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
from textblob import TextBlob

# Load datasets

"""
The list of features that we will need:

1. Metadata
  *   id
  *   genre
  *   artist_pop
  *   track_pop

2. Audio
  *   Mood: Danceability, Valence, Energy, Tempo
  *   Properties: Loudness, Speechiness, Instrumentalness
  *   Context: Liveness, Acousticness
  *   metadata: key,mode
3. Text
  *   track_name

"""

pd.options.mode.chained_assignment = None

class Recommendations(object):
  path = None
  df = None
  songDf = None
  final_vector = None
  
  def __init__(self, path):
    self.path = path
    self.df = pd.read_csv(path)

  def load_and_clean_dataset (self):
    self.df.drop(columns = ['Unnamed: 0.1', 'Unnamed: 0'], inplace = True)

    self.df['artist_songs'] = self.df.apply(lambda row : row['artist_name'] + row['track_name'], axis = 1)
    
    self.songDf = self.df.drop_duplicates('artist_songs')

  def select_cols (self):
    self.songDf = self.df[['artist_name','id','track_name','danceability', 'energy', 'key', 'loudness', 'mode',
         'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', "artist_pop", "genres", "track_pop"]]

  def split_genres (self):
    # print(df.dtypes)
    self.songDf['genre_list'] = self.songDf['genres'].apply(lambda x : x.split(" ") if type(x) != 'float' else print('',end=''))
    
  def playlist_preprocessor (self):
    self.load_and_clean_dataset()
    self.select_cols()
    self.split_genres()
  
  def getSubjectivity (self, text):
    return TextBlob(text).sentiment.subjectivity

  def getPolarity(self, text):
    return TextBlob(text).sentiment.polarity

  def getAnalysis (self, score, task="polarity"):
    if task == "subjectivity":
      if score < 1/3: return "low"
      elif score > 1/3: return "high"
      else: return "medium"

    else:
      if score < -0.13: return "Negative"
      elif score > 0.13: return "Positive"
      else: return "Neutral"

  def sentiment_analysis (self, df, text_col):
    df['subjectivity'] = df[text_col].apply(self.getSubjectivity).apply(lambda x : self.getAnalysis(x, "subjectivity"))
    df['polarity'] = df[text_col].apply(self.getPolarity).apply(lambda x : self.getAnalysis(x))
    
    return df

  def ohe_prep(self, df, column, new_name):
    tf_df = pd.get_dummies(df[column])
    features = tf_df.columns
    tf_df.columns = [new_name + "|" + str(i) for i in features]
    tf_df.reset_index(drop = True, inplace = True)
    
    return tf_df

  def create_feature (self, float_cols):
    scaler = MinMaxScaler()
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(self.songDf['genre_list'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names_out()]
    genre_df.drop(columns='genre|unknown')
    genre_df.reset_index(drop=True, inplace=True)
    
    # Sentiment analysis
    df = self.sentiment_analysis(self.songDf, "track_name")

    # One-hot encoding
    subject_ohe = self.ohe_prep(df, 'subjectivity', 'subject') * 0.3
    polar_ohe = self.ohe_prep(df, 'polarity', 'polar') * 0.5
    key_ohe = self.ohe_prep(df, 'key', 'key') * 0.5
    mode_ohe = self.ohe_prep(df, 'mode', 'mode') * 0.5
  
    # Normalization
    pop = df[["artist_pop", "track_pop"]].reset_index(drop=True)
    pop_scaled = pd.DataFrame(scaler.fit_transform(pop), columns = pop.columns)
  
    # Scale audio columns
    floats = df[float_cols].reset_index(drop = True)
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns)
  
    # Concatenate all features
    final = pd.concat([genre_df, floats_scaled, pop_scaled, subject_ohe, polar_ohe, key_ohe, mode_ohe], axis = 1)
  
    final["id"] = df["id"].values
    
    print("Feature set successfully built")
    self.final_vector = final
    return final

  def save_df (self, df, path):
    df.to_csv(path, index = False)

  def build_feature_set (self):
    float_cols = self.songDf.dtypes[self.songDf.dtypes == 'float64'].index.values
    complete_feature_set = self.create_feature(float_cols)
    self.save_df(complete_feature_set, "./data/complete_feature.csv")
    print("Feature set successfully built")

  def generate_playlist_feature (self, complete_feature_set, playlist_name = "Mom's playlist"):
    playlist_df = self.df[self.df['name'] == playlist_name]
    complete_feature_set_playlist = complete_feature_set[complete_feature_set['id'].isin(playlist_df['id'].values)]
    complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(playlist_df['id'].values)]
    complete_feature_set_playlist_final = complete_feature_set_playlist.drop(columns = "id")
    
    print("Playlist feature generated")
    return complete_feature_set_playlist_final.sum(axis = 0), complete_feature_set_nonplaylist

# feature_set_vector, nonplaylist_set = generate_playlist_feature(complete_feature_set, playlist_test)
# nonplaylist_set.head()
#
# feature_set_vector.head()

  def generate_playlist_recos (self, features, nonplaylist_features):
    non_playlist_df = self.df[self.df['id'].isin(nonplaylist_features['id'].values)]
    non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('id', axis = 1).values, features.values.reshape(1, -1))[:,0]
    non_playlist_df_top_40 = non_playlist_df.sort_values('sim', ascending = False).head(40)
  
    print("Recommendations generated for the given playlist")
    return non_playlist_df_top_40
  
  def get_recommendations (self):
    float_cols = self.songDf.dtypes[self.songDf.dtypes == 'float64'].index.values
    complete_feature_set = self.create_feature(float_cols)                                          # success
    feature_set_vector, nonplaylist_set = self.generate_playlist_feature(complete_feature_set)      # success
    recos_top40 = self.generate_playlist_recos(feature_set_vector, nonplaylist_set)
    return recos_top40

if __name__ == "__main__":
  pathtoDatasets = "../data/processed_data.csv"
  obj = Recommendations(pathtoDatasets)
  obj.playlist_preprocessor()
  print("Datasets loaded and cleaned for processing...")

  print("Now initiating recommendations system...")
  recoms = obj.get_recommendations()
  print(recoms.head())