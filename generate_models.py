import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
import os

def convert(obj):
    L = []
    try:
        for i in ast.literal_eval(obj):
            L.append(i['name'])
    except:
        pass
    return L

def convert3(obj):
    L = []
    counter = 0
    try:
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter += 1
            else:
                break
    except:
        pass
    return L

def fetch_director(obj):
    L = []
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
    except:
        pass
    return L

def run():
    print("Loading data...")
    if not os.path.exists('data/tmdb_5000_movies.csv') or not os.path.exists('data/tmdb_5000_credits.csv'):
        print("Data files not found in data/ directory.")
        return

    movies = pd.read_csv('data/tmdb_5000_movies.csv')
    credits = pd.read_csv('data/tmdb_5000_credits.csv')
    
    print("Merging data...")
    movies = movies.merge(credits, on='title')
    
    # Select columns including release_date and vote_average
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'release_date', 'vote_average']]
    
    print("Preprocessing...")
    movies.dropna(inplace=True)
    
    # Create year column
    movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year
    
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])
    
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    # Create new_df with necessary columns for the app
    new_df = movies[['movie_id', 'title', 'tags', 'year', 'vote_average']].copy()
    
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    
    print("Stemming...")
    ps = PorterStemmer()
    def stem(text):
        y = []
        for i in text.split():
            y.append(ps.stem(i))
        return " ".join(y)
        
    new_df['tags'] = new_df['tags'].apply(stem)
    
    print("Vectorizing...")
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    
    print("Calculating similarity...")
    similarity = cosine_similarity(vectors)
    
    print("Saving artifacts...")
    os.makedirs('artifacts', exist_ok=True)
    pickle.dump(new_df.to_dict(), open('artifacts/movie_dict.pkl', 'wb'))
    pickle.dump(similarity, open('artifacts/similarity.pkl', 'wb'))
    
    print("Done!")

if __name__ == "__main__":
    run()
