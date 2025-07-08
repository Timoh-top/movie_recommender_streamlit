import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import streamlit as st

# Title
st.title("🎬 Content-Based Movie Recommender")
st.write("Enter a movie title and get similar movie recommendations instantly.")

# Load Data
@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    
    def parse_features(x):
        try:
            items = ast.literal_eval(x)
            return ' '.join([i['name'] for i in items])
        except:
            return ''

    movies['genres_parsed'] = movies['genres'].apply(parse_features)
    movies['keywords_parsed'] = movies['keywords'].apply(parse_features)
    movies['combined_features'] = movies['overview'].fillna('') + ' ' + movies['genres_parsed'] + ' ' + movies['keywords_parsed']
    return movies

movies = load_data()

# TF-IDF
@st.cache_resource
def build_tfidf_matrix(movies):
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
    return tfidf_matrix

tfidf_matrix = build_tfidf_matrix(movies)

# Recommend function
def recommend_movies(title, top_n=10):
    idx = movies[movies['title'].str.lower() == title.lower()].index
    if len(idx) == 0:
        return movies[['title']].head(top_n)
    
    idx = idx[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]
    return movies.iloc[similar_indices][['title']]

# Streamlit UI
movie_name = st.text_input("Enter a movie title:", "")

if st.button("Get Recommendations"):
    if movie_name:
        results = recommend_movies(movie_name)
        if results is not None:
            st.write(f"Top Recommendations for **{movie_name.title()}**:")
            for idx, row in results.iterrows():
                st.write(f"- {row['title']}")
        else:
            st.error("Movie not found. Please check the title and try again.")
    else:
        st.warning("Please enter a movie title.")
