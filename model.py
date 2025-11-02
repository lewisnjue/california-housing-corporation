import joblib
import sys
import os
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
import gdown

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

_main_module = sys.modules.get('__main__')
if _main_module and not hasattr(_main_module, 'column_ratio'):
    _main_module.column_ratio = column_ratio
    _main_module.ratio_name = ratio_name
    _main_module.ClusterSimilarity = ClusterSimilarity

MODEL_FILE = "my_california_housing_model.pkl"
GOOGLE_DRIVE_FILE_ID = "1Mub1NiHQRAix3N7hcsN8DDWBKNN8FhX9"
GOOGLE_DRIVE_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"

def _download_model_if_needed():
    """Download model from Google Drive if it doesn't exist locally."""
    model_path = Path(MODEL_FILE)
    if not model_path.exists():
        try:
            print("Downloading model from Google Drive... This may take a few moments.")
            gdown.download(GOOGLE_DRIVE_URL, MODEL_FILE, quiet=True)
            print("Model downloaded successfully!")
        except Exception as e:
            raise Exception(
                f"Failed to download model from Google Drive: {str(e)}\n"
                f"Please ensure the Google Drive link is accessible: {GOOGLE_DRIVE_URL}"
            )

_model = None

def _load_model():
    global _model
    if _model is None:
        _download_model_if_needed()
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(
                f"Model file '{MODEL_FILE}' not found. "
                f"Please ensure you have trained the model or it can be downloaded from Google Drive."
            )
        _model = joblib.load(MODEL_FILE)
    return _model

def predict(data):
    model = _load_model()
    return model.predict(data)
