# Import necessary libraries
from scipy.sparse import issparse                     # Check if a matrix is sparse (used for performance optimization)
import streamlit as st                                # Streamlit for creating web app UI
import pandas as pd                                   # Pandas for data manipulation
import pickle                                         # Pickle for loading the saved TF-IDF model
from sklearn.metrics.pairwise import cosine_similarity  # For measuring similarity between movie vectors

# Set Streamlit page configuration
st.set_page_config(page_title='IMDB Movie Recommendation System Using Storylines', layout='wide')

# Load the TF-IDF model from a pickle file and cache it for performance
@st.cache_data
def load_pickle():
    try:
        with open(r"D:\Guvi_Project\IMDB Movie Recommendation System Using Storylines\data\tfidf.pkl", "rb") as file:
            model = pickle.load(file)  # Load the trained TF-IDF model
        return model
    except FileNotFoundError:
        st.error("TF-IDF model file not found. Please check the path.")
        st.stop()

# Load the main IMDB movie dataset and cache it
@st.cache_data
def load_data():
    try:
        df_data = pd.read_csv(r"D:\Guvi_Project\IMDB Movie Recommendation System Using Storylines\data\IMDb_Movies_story.csv")
        return df_data
    except FileNotFoundError:
        st.error("Movie data CSV file not found. Please check the path.")
        st.stop()

# Load the precomputed vector data for movies and cache it
@st.cache_data
def load_vector():
    try:
        df_vectors = pd.read_csv(r"D:\Guvi_Project\IMDB Movie Recommendation System Using Storylines\data\vector_data.csv")
        return df_vectors
    except FileNotFoundError:
        st.error("Vector data CSV file not found. Please check the path.")
        st.stop()

# Streamlit UI Elements
st.title("🎬 IMDB Movie Recommendation System Using Storylines")  # App title
st.subheader("Get recommendations based on movie storylines")     # App subtitle

# Load the cached model and datasets
tfidf_model = load_pickle()
df_data_csv = load_data()
df_vector_csv = load_vector()

# Input text area for the user to enter a movie storyline
user_input = st.text_area("Enter the storyline of a movie:")

# Function to generate recommendations based on user input
def recommendations(input_text):
    if input_text.strip():  # Check if the input is not empty
        input_text = input_text.lower()  # Convert input to lowercase for consistency
        input_vector = tfidf_model.transform([input_text])  # Transform the input text using TF-IDF

        # If the result is a sparse matrix, convert it to dense array
        if issparse(input_vector):
            input_vector = input_vector.toarray()

        # Compute cosine similarity between input and all movie vectors
        similarity_scores = cosine_similarity(df_vector_csv, input_vector)

        # Get indices of top 10 most similar movies
        top_indices = similarity_scores.flatten().argsort()[-10:][::-1]

        # Retrieve movie details from the dataset
        recommended_movies = df_data_csv.iloc[top_indices]
        return recommended_movies
    else:
        st.warning("Please enter a storyline to get recommendations.")  # Warn if input is empty
        return pd.DataFrame()

# Button to trigger recommendation generation
if st.button("🎥 Get Recommendations"):
    results = recommendations(user_input)
    if not results.empty:
        results = results[['Movie Name', 'Storyline']]  # Show only relevant columns
        results = results.reset_index(drop=True)        # Reset index for display
        results.index = results.index + 1               # Start index from 1
        st.subheader("Recommended Movies:")
        st.dataframe(results, use_container_width=True)  # Display results in a table
    else:
        st.warning("No recommendations available. Please try again with a different storyline.")  # Warn if no results found
