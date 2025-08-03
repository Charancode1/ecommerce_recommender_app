import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gdown
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --------------------------------------
# STEP 1: DOWNLOAD PRODUCT SIMILARITY FROM GOOGLE DRIVE
# --------------------------------------
file_id = "1yz9hrIN1VLOApuzr5Kiq_Dq39gu4p3GB"
download_url = f"https://drive.google.com/uc?id={file_id}"
similarity_file = "product_similarity.pkl"

if not os.path.exists(similarity_file):
    gdown.download(download_url, similarity_file, quiet=False)

# Load product similarity matrix
with open(similarity_file, 'rb') as f:
    similarity_matrix = pickle.load(f)

# --------------------------------------
# STEP 2: LOAD PRODUCT NAMES MAPPING
# --------------------------------------
# Ideally you should have a product list in same order as similarity matrix
# Example:
product_list = list(similarity_matrix.index)

# --------------------------------------
# STEP 3: LOAD RFM KMeans and Scaler MODELS
# --------------------------------------
with open("rfm_kmeans.pkl", 'rb') as f:
    kmeans_model = pickle.load(f)

with open("rfm_scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

# --------------------------------------
# 🎨 Streamlit UI
# --------------------------------------
st.set_page_config(page_title="E-commerce Recommender & Segmentation", layout="centered")

st.title("🛍️ E-commerce Recommendation System + Customer Segmentation")

# --------------------------------------
# 📦 MODULE 1: PRODUCT RECOMMENDATION
# --------------------------------------
st.header("📌 Product Recommendation")

product_input = st.text_input("Enter a Product Name:")
if st.button("🔍 Get Recommendations"):
    if product_input not in product_list:
        st.warning("Product not found in database. Try exact name from your dataset.")
    else:
        idx = product_list.index(product_input)
        sim_scores = list(enumerate(similarity_matrix.iloc[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_5 = [product_list[i[0]] for i in sim_scores[1:6]]

        st.subheader("🧠 Top 5 Similar Products:")
        for i, item in enumerate(top_5, 1):
            st.markdown(f"**{i}. {item}**")

# --------------------------------------
# 👥 MODULE 2: CUSTOMER SEGMENTATION
# --------------------------------------
st.header("🎯 Customer Segmentation")

r = st.number_input("Recency (days since last purchase)", min_value=0, step=1)
f = st.number_input("Frequency (total purchases)", min_value=0, step=1)
m = st.number_input("Monetary (total spend)", min_value=0.0, step=1.0)

if st.button("📊 Predict Cluster"):
    user_data = np.array([[r, f, m]])
    user_scaled = scaler.transform(user_data)
    cluster = kmeans_model.predict(user_scaled)[0]

    cluster_names = {
        0: "🟢 High-Value",
        1: "🔵 Regular",
        2: "🟡 Occasional",
        3: "🔴 At-Risk"
    }

    st.success(f"Customer is in Cluster: {cluster_names.get(cluster, 'Unknown')} (Cluster {cluster})")

# --------------------------------------
# Footer
# --------------------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")


