import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the preprocessed data
@st.cache_data
def load_data():
    return pd.read_csv('feature_engineered_restaurant_data.csv')


df = load_data()

# Select relevant features for content-based filtering
content_features = ['StarRating', 'NumberOfReviews', 'Weighted_Rating', 'Sentiment_Rating_Interaction'] + \
                  [col for col in df.columns if 'Style_' in col]

# Impute missing values with 0
df[content_features] = df[content_features].fillna(0)

# Compute the cosine similarity matrix
content_similarity_matrix = cosine_similarity(df[content_features])

# Function to recommend restaurants based on content similarity
def recommend_restaurant_content(restaurant_name, num_recommendations=5):
    try:
        # Find the index of the restaurant
        idx = df[df['RestaurantName'] == restaurant_name].index[0]
    except IndexError:
        st.error(f"Restaurant '{restaurant_name}' not found in the dataset.")
        return pd.DataFrame()  # Return an empty DataFrame if the restaurant is not found

    # Get the similarity scores
    similarity_scores = list(enumerate(content_similarity_matrix[idx]))

    # Sort restaurants by similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get indices of the most similar restaurants
    similar_restaurant_indices = [i[0] for i in similarity_scores[1:num_recommendations + 1]]

    # Return the recommended restaurants
    return df.iloc[similar_restaurant_indices][['RestaurantName', 'StarRating', 'NumberOfReviews', 'Style']]

# Streamlit Application
st.title("Restaurant Recommendation System")
st.write("This application recommends restaurants based on content similarity.")

# User input: Select a restaurant
restaurant_list = df['RestaurantName'].unique()
selected_restaurant = st.selectbox("Select a restaurant:", restaurant_list)

# User input: Number of recommendations
num_recommendations = st.slider("Number of recommendations:", min_value=1, max_value=10, value=5)

# Generate recommendations
if st.button("Show Recommendations"):
    st.write(f"Recommendations for **{selected_restaurant}**:")
    recommendations = recommend_restaurant_content(selected_restaurant, num_recommendations)

    if not recommendations.empty:
        for idx, row in recommendations.iterrows():
            st.write(f"- **{row['RestaurantName']}**")
            st.write(f"  - Star Rating: {row['StarRating']}")
            st.write(f"  - Number of Reviews: {row['NumberOfReviews']}")
            st.write(f"  - Style: {row['Style']}")
    else:
        st.warning("No recommendations available.")
