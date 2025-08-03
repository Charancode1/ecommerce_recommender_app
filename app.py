import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# --- Set up Drive file download ---
@st.cache_data
def load_similarity_matrix():
    url = "https://drive.google.com/uc?id=1yz9hrIN1VLOApuzr5Kiq_Dq39gu4p3GB"
    output_path = "product_similarity.pkl"
    
    # Download the file
    response = requests.get(url)
    with open(output_path, "wb") as f:
        f.write(response.content)

    return joblib.load(output_path)

# --- Load data ---
@st.cache_data
def load_product_data():
    # Sample product list (replace with actual if available)
    return pd.DataFrame({
        "product_name": [
            "Laptop Sleeve", "Bluetooth Headphones", "Gaming Mouse", 
            "LED Monitor", "Portable Charger"
        ]
    })

# --- Recommender Function ---
def recommend_products(selected_product, product_list, similarity_matrix):
    idx = product_list.index(selected_product)
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended = [product_list[i[0]] for i in similarity_scores[1:6]]
    return recommended

# --- Streamlit UI ---
st.title("🛍️ E-Commerce Product Recommender")
product_df = load_product_data()
similarity_matrix = load_similarity_matrix()

selected = st.selectbox("Choose a product", product_df['product_name'].values)

if st.button("Recommend Similar Products"):
    recommendations = recommend_products(selected, list(product_df['product_name']), similarity_matrix)
    st.subheader("You may also like:")
    for rec in recommendations:
        st.write(f"✅ {rec}")




