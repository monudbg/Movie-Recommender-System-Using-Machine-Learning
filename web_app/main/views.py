import pickle
import pandas as pd
import requests
import os
from django.shortcuts import render
from django.conf import settings

# Construct absolute paths to artifacts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'artifacts')

# Load models safely
try:
    movies_dict = pickle.load(open(os.path.join(ARTIFACTS_DIR, 'movie_dict.pkl'), 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open(os.path.join(ARTIFACTS_DIR, 'similarity.pkl'), 'rb'))
except FileNotFoundError:
    movies = pd.DataFrame()
    similarity = None
    print("Warning: Artifacts not found. Please run generate_models.py in the root directory.")

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    try:
        data = requests.get(url, timeout=5)
        data.raise_for_status()
        data = data.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
    except Exception as e:
        print(f"Error fetching poster: {e}")
    return "https://placehold.co/500x750/333/FFFFFF?text=No+Poster"

def index(request):
    movie_list = movies['title'].values if not movies.empty else []
    return render(request, 'main/index.html', {'movie_list': movie_list})

def recommend(request):
    if request.method == 'POST':
        selected_movie = request.POST.get('selected_movie')
        
        if movies.empty or similarity is None:
            return render(request, 'main/index.html', {
                'movie_list': movies['title'].values if not movies.empty else [],
                'error': 'Models not loaded. Please ensure artifacts are generated.'
            })

        try:
            index = movies[movies['title'] == selected_movie].index[0]
            distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
            
            recommendations = []
            for i in distances[1:6]:
                movie_id = movies.iloc[i[0]].movie_id
                title = movies.iloc[i[0]].title
                year = movies.iloc[i[0]].year
                rating = movies.iloc[i[0]].vote_average
                
                poster = fetch_poster(movie_id)
                recommendations.append({
                    'title': title,
                    'poster': poster,
                    'year': int(year) if pd.notna(year) else 'N/A',
                    'rating': rating
                })
            
            return render(request, 'main/index.html', {
                'movie_list': movies['title'].values,
                'selected_movie': selected_movie,
                'recommendations': recommendations
            })
            
        except IndexError:
            return render(request, 'main/index.html', {
                'movie_list': movies['title'].values,
                'error': 'Movie not found.'
            })
    
    return index(request)
