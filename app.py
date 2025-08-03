import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load models
with open("rfm_kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("rfm_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("product_similarity.pkl", "rb") as f:
    product_sim = pickle.load(f)

# --- UI ---
st.title("🛍️ Shopper Spectrum - E-Commerce Recommender App")
st.markdown("---")

tab1, tab2 = st.tabs(["🎯 Product Recommender", "🔍 Customer Segmentation"])

# --- Recommender ---
with tab1:
    st.subheader("🎯 Product Recommendation")
    product_name = st.text_input("Enter Product Name")
    
    if st.button("Get Recommendations"):
        if product_name in product_sim.index:
            sim_scores = product_sim[product_name].sort_values(ascending=False)[1:6]
            st.success("Top 5 Recommended Products:")
            for i, prod in enumerate(sim_scores.index, 1):
                st.markdown(f"**{i}.** {prod}")
        else:
            st.error("Product not found. Try another one.")

# --- Segmentation ---
with tab2:
    st.subheader("🔍 Predict Customer Segment")

    recency = st.number_input("Recency (days)", min_value=0)
    frequency = st.number_input("Frequency (purchases)", min_value=0)
    monetary = st.number_input("Monetary (₹ spent)", min_value=0.0)

    if st.button("Predict Cluster"):
        rfm_input = scaler.transform([[recency, frequency, monetary]])
        cluster = kmeans.predict(rfm_input)[0]

        # Mapping clusters to labels (edit as per your kmeans profile)
        label_map = {
            0: "High-Value",
            1: "Regular",
            2: "Occasional",
            3: "At-Risk",
            4: "Others"
        }

        st.success(f"Predicted Cluster: **{label_map.get(cluster, 'Unknown')}**")
