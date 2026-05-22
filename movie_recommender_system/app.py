import streamlit as st
import pickle
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Page Configuration
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    color: white;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #AAAAAA;
    font-size: 18px;
    margin-bottom: 40px;
}

.stButton>button {
    width: 100%;
    background-color: transparent;
    color: white;
    font-size: 16px;
    font-weight: 500;
    padding: 12px 20px;
    border-radius: 8px;
    border: 1px solid #3a3a3a;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    border: 1px solid #E50914;
    color: #E50914;
    background-color: rgba(229, 9, 20, 0.08);
}

.movie-card {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 15px;
    color: white;
    font-size: 20px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
}

</style>
""", unsafe_allow_html=True)


# Load Data
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)


# Recommendation Function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies


# UI
st.markdown('<div class="title"> Movie Recommender System</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Discover movies similar to your favorite ones</div>',
    unsafe_allow_html=True
)

selected_movie_name = st.selectbox(
    "Select A Movie",
    movies['title'].values
)

if st.button('Recommend Movies'):

    recommendations = recommend(selected_movie_name)

    st.success("Top Recommended Movies")

    for movie in recommendations:
        st.markdown(
            f'<div class="movie-card">{movie}</div>',
            unsafe_allow_html=True
        )