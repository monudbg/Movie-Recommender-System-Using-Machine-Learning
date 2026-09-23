"""
Movie Recommendation Engine — Data Processing & Model Generation
==================================================================

This script handles data ingestion, feature engineering, Natural Language
Processing (NLP), vector space embedding, and cosine similarity matrix computation
for the TMDB 5000 Movies dataset.

Pipeline Steps:
1. Load raw CSVs: tmdb_5000_movies.csv & tmdb_5000_credits.csv.
2. Merge datasets on movie title.
3. Parse complex stringified JSON objects (genres, keywords, cast, crew).
4. Extract Director name from crew JSON and top 3 actors from cast JSON.
5. Combine overview, genres, keywords, cast, and crew into unified 'tags' feature.
6. Apply Porter Stemmer for NLP stemming (e.g., 'actions', 'acting' -> 'act').
7. Build 5,000-feature Bag-of-Words space using CountVectorizer.
8. Compute 4800x4800 Cosine Similarity Matrix.
9. Precalculate top-10 recommended indices map to eliminate live runtime latency.
10. Serialize outputs into `artifacts/` pickle files.

Developer Notes:
- Keep max_features=5000 in CountVectorizer to balance accuracy and memory footprint.
- Precalculating top-10 indices reduces web server query time from 1.5s to <50ms.
"""

import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
import os

# ---------------------------------------------------------------------------
# AST Data Extraction Helpers
# ---------------------------------------------------------------------------

def convert(obj):
    """Safely parse a stringified JSON list of dicts and extract 'name' fields.
    
    Example input: "[{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]"
    Returns: ['Action', 'Adventure']
    """
    L = []
    try:
        for i in ast.literal_eval(obj):
            L.append(i['name'])
    except Exception:
        pass
    return L


def convert3(obj):
    """Extract top 3 cast actor names from stringified cast JSON list.
    
    Args:
        obj (str): Stringified JSON representing movie cast members.
        
    Returns:
        list[str]: Names of top 3 lead actors.
    """
    L = []
    counter = 0
    try:
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter += 1
            else:
                break
    except Exception:
        pass
    return L


def fetch_director(obj):
    """Scan crew JSON list to find and extract the Director's name.
    
    Args:
        obj (str): Stringified JSON representing movie crew members.
        
    Returns:
        list[str]: Single-element list containing Director's name (or empty list).
    """
    L = []
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
    except Exception:
        pass
    return L


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------

def run():
    """Execute complete model training and artifact generation pipeline."""
    print("[1/6] Loading TMDB datasets...")
    if not os.path.exists('data/tmdb_5000_movies.csv') or not os.path.exists('data/tmdb_5000_credits.csv'):
        print("ERROR: Data files not found in data/ directory.")
        return

    movies = pd.read_csv('data/tmdb_5000_movies.csv')
    credits = pd.read_csv('data/tmdb_5000_credits.csv')
    
    print("[2/6] Merging movies and credits dataframes...")
    movies = movies.merge(credits, on='title')
    
    # Select relevant columns for feature building and web UI
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'release_date', 'vote_average']]
    
    print("[3/6] Cleaning missing values and feature engineering...")
    movies.dropna(inplace=True)
    
    # Extract release year from date column
    movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year
    
    # Apply AST parsing helpers
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    
    # Convert overview text into word tokens
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    
    # Remove whitespace between multi-word names (e.g., 'Sam Worthington' -> 'SamWorthington')
    # This prevents the vectorizer from confusing two people sharing a first name
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])
    
    # Concatenate all metadata into a single 'tags' token list
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    # Build clean output DataFrame
    new_df = movies[['movie_id', 'title', 'tags', 'year', 'vote_average']].copy()
    
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    
    print("[4/6] Performing NLP Stemming using PorterStemmer...")
    ps = PorterStemmer()
    def stem(text):
        y = []
        for i in text.split():
            y.append(ps.stem(i))
        return " ".join(y)
        
    new_df['tags'] = new_df['tags'].apply(stem)
    
    print("[5/6] Vectorizing text with CountVectorizer (max_features=5000, English stop_words)...")
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    
    print("[6/6] Calculating Cosine Similarity Matrix and precomputing top-10 maps...")
    similarity = cosine_similarity(vectors)
    
    recommendations_dict = {}
    for idx in range(len(new_df)):
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
        # Top 10 recommendations (excluding self at index 0)
        rec_indices = [int(i[0]) for i in distances[1:11]]
        recommendations_dict[int(idx)] = rec_indices

    print("Saving serialized model artifacts to artifacts/ directory...")
    os.makedirs('artifacts', exist_ok=True)
    pickle.dump(new_df.to_dict(), open('artifacts/movie_dict.pkl', 'wb'))
    pickle.dump(recommendations_dict, open('artifacts/recommendations.pkl', 'wb'))
    pickle.dump(similarity, open('artifacts/similarity.pkl', 'wb'))
    
    print("[Done] Model artifacts generated successfully.")

if __name__ == "__main__":
    run()
