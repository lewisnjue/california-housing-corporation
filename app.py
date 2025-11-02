import streamlit as st
import pandas as pd
import numpy as np
import sys
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

sys.modules['__main__'].column_ratio = column_ratio
sys.modules['__main__'].ratio_name = ratio_name
sys.modules['__main__'].ClusterSimilarity = ClusterSimilarity

from model import predict
from visualizations import create_all_visualizations
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="wide"
)

page = st.sidebar.selectbox("Select Page", ["Model Prediction", "Visualizations"])

if page == "Model Prediction":
    st.title("🏠 California Housing Price Prediction")
    st.markdown("Enter the details below to predict the median house value:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        longitude = st.number_input("Longitude", value=-122.23, min_value=-124.0, max_value=-114.0, step=0.01)
        latitude = st.number_input("Latitude", value=37.88, min_value=32.0, max_value=42.0, step=0.01)
        housing_median_age = st.number_input("Housing Median Age", value=52.0, min_value=0.0, step=1.0)
        total_rooms = st.number_input("Total Rooms", value=2000.0, min_value=0.0, step=100.0)
        total_bedrooms = st.number_input("Total Bedrooms", value=400.0, min_value=0.0, step=10.0)
    
    with col2:
        population = st.number_input("Population", value=1500.0, min_value=0.0, step=100.0)
        households = st.number_input("Households", value=500.0, min_value=0.0, step=10.0)
        median_income = st.number_input("Median Income", value=3.87, min_value=0.0, step=0.1)
        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            options=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
        )
    
    if st.button("Predict House Value", type="primary"):
        input_data = pd.DataFrame({
            'longitude': [longitude],
            'latitude': [latitude],
            'housing_median_age': [housing_median_age],
            'total_rooms': [total_rooms],
            'total_bedrooms': [total_bedrooms],
            'population': [population],
            'households': [households],
            'median_income': [median_income],
            'ocean_proximity': [ocean_proximity]
        })
        
        try:
            prediction = predict(input_data)
            predicted_value = prediction[0]
            
            st.success(f"### Predicted Median House Value: ${predicted_value:,.2f}")
            
            st.markdown("---")
            st.markdown("### Input Summary")
            st.dataframe(input_data, use_container_width=True)
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

elif page == "Visualizations":
    st.title("📊 California Housing Data Visualizations")
    st.markdown("Explore various visualizations of the California housing dataset")
    
    if st.button("Generate Visualizations", type="primary"):
        with st.spinner("Loading data and generating visualizations..."):
            figs = create_all_visualizations()
        
        st.markdown("### 1. Feature Histograms")
        st.pyplot(figs['histograms'])
        
        st.markdown("### 2. Income Category Distribution")
        st.pyplot(figs['income_category'])
        
        st.markdown("### 3. Geographic Distribution - Basic")
        st.pyplot(figs['geographic_basic'])
        
        st.markdown("### 4. Geographic Distribution - With Transparency")
        st.pyplot(figs['geographic_alpha'])
        
        st.markdown("### 5. Geographic Distribution - With House Values")
        st.pyplot(figs['geographic_value'])
        
        st.markdown("### 6. Scatter Matrix")
        st.pyplot(figs['scatter_matrix'])
        
        st.markdown("### 7. Median Income vs House Value")
        st.pyplot(figs['income_vs_value'])
        
        st.markdown("### 8. Population Distribution")
        st.pyplot(figs['population_dist'])
        
        st.markdown("### 9. Housing Age Similarity")
        st.pyplot(figs['age_similarity'])
        
        st.markdown("### 10. Cluster Similarity Map")
        st.pyplot(figs['cluster_similarity'])

