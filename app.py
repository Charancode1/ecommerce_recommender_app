import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
kmeans = joblib.load("rfm_kmeans.pkl")
scaler = joblib.load("rfm_scaler.pkl")

# Streamlit UI
st.set_page_config(page_title="RFM Customer Segment Predictor", layout="centered")

st.title("🛍️ RFM Customer Segment Predictor")
st.write("Provide your customer data below to predict the customer segment using a trained KMeans model.")

# Input fields
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=1000, value=30)
frequency = st.number_input("Frequency (number of purchases)", min_value=0, max_value=1000, value=10)
monetary = st.number_input("Monetary Value (total spent)", min_value=0.0, max_value=100000.0, value=500.0)

if st.button("Predict Segment"):
    try:
        input_data = pd.DataFrame([[recency, frequency, monetary]], columns=["Recency", "Frequency", "Monetary"])
        scaled_data = scaler.transform(input_data)
        segment = kmeans.predict(scaled_data)[0]
        
        st.success(f"✅ Predicted Customer Segment: **Segment {segment}**")
        
        # Optional: Add descriptions for each segment if known
        segment_descriptions = {
            0: "💎 Loyal customers who buy frequently and spend a lot.",
            1: "🛍️ Average customers with moderate activity.",
            2: "⏸️ Inactive or low-value customers."
        }
        st.info(segment_descriptions.get(segment, "Segment description not available."))

    except Exception as e:
        st.error(f"Something went wrong during prediction. Error: {str(e)}")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit.")


