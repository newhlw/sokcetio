import os
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

class OMDbRecommender:
    def __init__(self):
        self.api_key = os.getenv('OMDB_API_KEY')
        self.base_url = "http://www.omdbapi.com/"
        self.model = None
        self.movie_user_mat = None
        self.mapper = {}
        self.movies_df = pd.DataFrame()
        self.ratings_df = pd.DataFrame()
        self.initialized = False
        
    def initialize(self):
        """Initialize all components in the correct order"""
        if not self.initialized:
            self.fetch_popular_movies()
            self.prepare_data()
            self.train_model()
            self.initialized = True
        
    def fetch_popular_movies(self, n_movies=100):
        """Fetch popular movies by searching common terms"""
        search_terms = ['action', 'comedy', 'drama', 'sci-fi', 'adventure']
        movies = []
        
        for term in search_terms:
            params = {
                'apikey': self.api_key,
                's': term,
                'type': 'movie'
            }
            response = requests.get(self.base_url, params=params)
            if response.status_code == 200:
                movies.extend(response.json().get('Search', []))
        
        # Create DataFrame with unique movies
        unique_movies = {m['imdbID']: m for m in movies}.values()
        self.movies_df = pd.DataFrame([{
            'imdbID': m['imdbID'],
            'title': m['Title'],
            'year': m['Year'],
            'type': m['Type']
        } for m in unique_movies])
        
        # Create mock ratings (in production, use real user ratings)
        np.random.seed(42)
        user_ids = np.random.randint(1, 100, size=1000)
        movie_ids = np.random.choice(self.movies_df['imdbID'], size=1000)
        ratings = np.random.uniform(1, 5, size=1000).round(1)
        
        self.ratings_df = pd.DataFrame({
            'userId': user_ids,
            'imdbID': movie_ids,
            'rating': ratings
        })
        
    def prepare_data(self):
        """Prepare data for the recommender"""
        # Merge movies and ratings
        movie_ratings = pd.merge(self.movies_df, self.ratings_df, on='imdbID')
        
        # Create user-item matrix
        movie_ratings_pivot = movie_ratings.pivot_table(
            index='title', columns='userId', values='rating').fillna(0)
            
        # Convert to sparse matrix
        self.movie_user_mat = csr_matrix(movie_ratings_pivot.values)
        
        # Create title to index mapper
        self.mapper = {
            title: index for index, title in enumerate(movie_ratings_pivot.index)
        }
        
    def train_model(self, n_neighbors=5):
        """Train the KNN model"""
        self.model = NearestNeighbors(
            n_neighbors=n_neighbors, 
            metric='cosine', 
            algorithm='brute')
        self.model.fit(self.movie_user_mat)
        
    def recommend_movies(self, movie_title, n_recommendations=5):
        """Get movie recommendations"""
        if not self.initialized:
            self.initialize()
            
        idx = self.mapper.get(movie_title, None)
        if idx is None:
            return []
            
        # Find similar movies
        distances, indices = self.model.kneighbors(
            self.movie_user_mat[idx], 
            n_neighbors=n_recommendations+1)
            
        # Process results
        raw_recommends = sorted(
            list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
            key=lambda x: x[1])[:0:-1]
            
        reverse_mapper = {v: k for k, v in self.mapper.items()}
        return [(reverse_mapper[i], 1 - d) for i, d in raw_recommends]
    
    def get_movie_titles(self):
        """Get list of available movie titles"""
        if not self.initialized:
            self.initialize()
        return list(self.mapper.keys())
    
    def get_movie_details(self, title):
        """Get detailed information about a movie"""
        params = {
            'apikey': self.api_key,
            't': title
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            return response.json()
        return None

# Initialize recommender
recommender = OMDbRecommender()

@app.route('/')
def home():
    return render_template('index.html', movies=recommender.get_movie_titles())

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    movie_title = request.form.get('movie_title')
    if not movie_title:
        return render_template('index.html', error="Please select a movie")
    
    recommendations = recommender.recommend_movies(movie_title)
    movie_details = recommender.get_movie_details(movie_title)
    
    return render_template(
        'index.html',
        movies=recommender.get_movie_titles(),
        recommendations=recommendations,
        selected_movie=movie_title,
        movie_details=movie_details
    )

@app.route('/search', methods=['GET'])
def search_movies():
    query = request.args.get('query', '')
    params = {
        'apikey': os.getenv('OMDB_API_KEY'),
        's': query,
        'type': 'movie'
    }
    response = requests.get("http://www.omdbapi.com/", params=params)
    if response.status_code == 200:
        results = [item['Title'] for item in response.json().get('Search', [])]
        return jsonify(results)
    return jsonify([])

if __name__ == '__main__':
    # Initialize the recommender before running the app
    recommender.initialize()
    app.run(debug=True)