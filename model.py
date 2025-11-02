import joblib
import sys
import numpy as np
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

_main_module = sys.modules.get('__main__')
if _main_module and not hasattr(_main_module, 'column_ratio'):
    _main_module.column_ratio = column_ratio
    _main_module.ratio_name = ratio_name
    _main_module.ClusterSimilarity = ClusterSimilarity

_model = None

def _load_model():
    global _model
    if _model is None:
        _model = joblib.load("my_california_housing_model.pkl")
    return _model

def predict(data):
    model = _load_model()
    return model.predict(data)
