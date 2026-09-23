"""
Movie Recommender System — Django View Controllers & API Service
================================================================

This module implements the primary view handlers for the recommendation app.

Key Architectural Responsibilities:
1. Artifact Loading: Safe unpickling of `movie_dict.pkl` and `recommendations.pkl`.
2. Async Poster Fetching: Queries TMDB API using `ThreadPoolExecutor` for parallel execution.
3. Multi-Level Caching: Uses Django File/Memory Cache for poster URLs (30 days) and full recommendation sets (24 hours).
4. Watch Provider Lookup: Queries TMDB watch provider endpoints for streaming links.

Developer Instructions:
- Ensure TMDB_API_KEY is active.
- To clear poster cache if images break, run `python manage.py clear_cache` or flush django cache directory.
"""

import pickle
import pandas as pd
import requests
import os
from django.shortcuts import render
from django.conf import settings
from django.core.cache import cache
from concurrent.futures import ThreadPoolExecutor

# TMDB API Key for fetching poster images and streaming metadata
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

# Construct absolute paths to model artifacts directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "artifacts")

# Safe unpickling of model data structures
try:
    movies_dict = pickle.load(open(os.path.join(ARTIFACTS_DIR, "movie_dict.pkl"), "rb"))
    movies = pd.DataFrame(movies_dict)
    recommendations_data = pickle.load(open(os.path.join(ARTIFACTS_DIR, "recommendations.pkl"), "rb"))
except FileNotFoundError:
    movies = pd.DataFrame()
    recommendations_data = None
    print(
        "WARNING: Artifacts not found in artifacts/ directory. "
        "Please execute 'python generate_models.py' in the repository root."
    )


# ---------------------------------------------------------------------------
# TMDB API Helpers with Django Caching
# ---------------------------------------------------------------------------

def fetch_poster(movie_id: int) -> str:
    """Fetch movie poster artwork URL from TMDB API v3.
    
    Implements a 30-day Django cache strategy to minimize external network requests.
    
    Args:
        movie_id (int): TMDB movie identifier.
        
    Returns:
        str: Fully constructed poster image URL.
    """
    cache_key = f"poster_{movie_id}"
    cached_poster = cache.get(cache_key)
    if cached_poster:
        return cached_poster

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    try:
        data = requests.get(url, timeout=3)
        data.raise_for_status()
        data = data.json()
        poster_path = data.get("poster_path")
        if poster_path:
            poster_url = "https://image.tmdb.org/t/p/w500/" + poster_path
            cache.set(cache_key, poster_url, timeout=86400 * 30)  # Cache for 30 days
            return poster_url
    except Exception as e:
        print(f"Error fetching poster for movie {movie_id}: {e}")

    default_poster = "https://placehold.co/500x750/333/FFFFFF?text=No+Poster"
    cache.set(cache_key, default_poster, timeout=3600)  # Cache failure fallback for 1 hour
    return default_poster


def fetch_watch_provider(movie_id: int) -> str:
    """Return a US streaming provider URL for a given TMDB movie ID.
    
    Caches successful link lookups for 30 days.
    
    Args:
        movie_id (int): TMDB movie identifier.
        
    Returns:
        str: Direct streaming provider URL (or '#' fallback).
    """
    cache_key = f"watch_{movie_id}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers?api_key={TMDB_API_KEY}"
    try:
        resp = requests.get(url, timeout=3)
        resp.raise_for_status()
        data = resp.json()
        providers = data.get("results", {}).get("US", {})
        flatrate = providers.get("flatrate")
        if flatrate:
            link = flatrate[0].get("link")
            if link:
                cache.set(cache_key, link, timeout=86400 * 30)  # Cache 30 days
                return link
    except Exception as e:
        print(f"Watch provider lookup failed for movie {movie_id}: {e}")

    fallback = "#"
    cache.set(cache_key, fallback, timeout=3600)  # Cache failure 1 hour
    return fallback


def get_movie_rec_details(i: int) -> dict:
    """Extract metadata, poster image, and watch link for a given movie index.
    
    Args:
        i (int): Row index in the pandas DataFrame.
        
    Returns:
        dict: Movie recommendation card data dictionary.
    """
    movie_id = movies.iloc[i].movie_id
    title = movies.iloc[i].title
    year = movies.iloc[i].year
    rating = movies.iloc[i].vote_average
    poster = fetch_poster(movie_id)
    provider_url = fetch_watch_provider(movie_id)
    return {
        "title": title,
        "poster": poster,
        "year": int(year) if pd.notna(year) else "N/A",
        "rating": rating,
        "watch_url": provider_url,
    }


# ---------------------------------------------------------------------------
# View Controllers
# ---------------------------------------------------------------------------

def index(request):
    """Render main homepage with dropdown list of all available movies."""
    movie_list = movies["title"].values if not movies.empty else []
    return render(request, "main/index.html", {"movie_list": movie_list})


def recommend(request):
    """Process user movie selection POST request and return top 5 recommendations.

    Optimization Flow:
    1. Check if the full recommendation card set for the selected movie is in cache.
    2. If cache miss, fetch precomputed top 5 recommendation indices from recommendations_data.
    3. Use ThreadPoolExecutor to fetch poster images and watch links in parallel (5 threads).
    4. Cache the resulting recommendations list for 24 hours.
    """
    if request.method == "POST":
        selected_movie = request.POST.get("selected_movie")

        if movies.empty or recommendations_data is None:
            return render(
                request,
                "main/index.html",
                {
                    "movie_list": movies["title"].values if not movies.empty else [],
                    "error": "Models not loaded. Please ensure artifacts are generated by running generate_models.py.",
                },
            )

        # Attempt to retrieve cached recommendation list
        rec_cache_key = f"recs_{selected_movie.replace(' ', '_').lower()}"
        cached_recommendations = cache.get(rec_cache_key)
        if cached_recommendations:
            return render(
                request,
                "main/index.html",
                {
                    "movie_list": movies["title"].values,
                    "selected_movie": selected_movie,
                    "recommendations": cached_recommendations,
                },
            )

        try:
            # Find the row index of the user-selected movie
            movie_idx = movies[movies["title"] == selected_movie].index[0]
            
            # Fetch top 5 precalculated recommendation indices
            rec_indices = recommendations_data.get(int(movie_idx), [])[:5]

            # Parallelize poster and metadata fetching using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=5) as executor:
                recommendations = list(executor.map(get_movie_rec_details, rec_indices))

            # Store result in Django cache for 24 hours
            cache.set(rec_cache_key, recommendations, timeout=86400)

            return render(
                request,
                "main/index.html",
                {
                    "movie_list": movies["title"].values,
                    "selected_movie": selected_movie,
                    "recommendations": recommendations,
                },
            )

        except IndexError:
            return render(
                request,
                "main/index.html",
                {"movie_list": movies["title"].values, "error": "Movie title not found in dataset."},
            )

    return index(request)
