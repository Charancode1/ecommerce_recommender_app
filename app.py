import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
from io import BytesIO

# ---- HELPER FUNCTION TO LOAD LARGE PICKLE FILE FROM GOOGLE DRIVE ----
@st.cache_data(show_spinner=True)
def download_pickle_from_drive():
    file_id = "1yz9hrIN1VLOApuzr5Kiq_Dq39gu4p3GB"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return joblib.load(BytesIO(response.content))
    else:
        st.error("Failed to download the similarity file.")
        return None

# ---- LOAD FILES ----
st.set_page_config(page_title="E-Commerce Recommender App", layout="centered")

st.title("🛒 E-Commerce Recommender App")

similarity_df = download_pickle_from_drive()

scaler = joblib.load("rfm_scaler.pkl")
kmeans = joblib.load("rfm_kmeans.pkl")

# ---- PRODUCT RECOMMENDATION ----
st.header("🎯 Product Recommendation")
product_input = st.text_input("Enter Product Name")

if st.button("🔍 Get Recommendations"):
    if product_input not in similarity_df.columns:
        st.warning("Product not found. Please try a different name.")
    else:
        top_5 = similarity_df[product_input].sort_values(ascending=False)[1:6]
        st.success(f"Top 5 recommendations for **{product_input}**:")
        for i, prod in enumerate(top_5.index, 1):
            st.markdown(f"**{i}. {prod}**")

# ---- CUSTOMER SEGMENTATION ----
st.header("🎯 Customer Segmentation (RFM)")

recency = st.number_input("Recency (in days)", min_value=0, value=10)
frequency = st.number_input("Frequency (No. of Purchases)", min_value=0, value=5)
monetary = st.number_input("Monetary (Total Spend)", min_value=0, value=100)

if st.button("🎯 Predict Customer Cluster"):
    user_input = pd.DataFrame([[recency, frequency, monetary]], columns=["Recency", "Frequency", "Monetary"])
    user_scaled = scaler.transform(user_input)
    cluster = kmeans.predict(user_scaled)[0]
    
    cluster_labels = {
        0: "High-Value",
        1: "Regular",
        2: "Occasional",
        3: "At-Risk"
    }

    label = cluster_labels.get(cluster, "Unknown")
    st.success(f"Customer Segment: **{label}** (Cluster {cluster})")



