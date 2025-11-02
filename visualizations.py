import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import tarfile
import urllib.request
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from pandas.plotting import scatter_matrix

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
        housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

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

def create_all_visualizations():
    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    housing = load_housing_data()

    figs = {}
    
    fig1, axes = plt.subplots(3, 3, figsize=(12, 8))
    axes = axes.flatten()
    numeric_cols = housing.select_dtypes(include=[np.number]).columns
    for i, col in enumerate(numeric_cols[:9]):
        housing[col].hist(bins=50, ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
    figs['histograms'] = fig1

    fig2, ax = plt.subplots(figsize=(10, 6))
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True, ax=ax)
    ax.set_xlabel("Income category")
    ax.set_ylabel("Number of districts")
    plt.tight_layout()
    figs['income_category'] = fig2

    fig3, ax = plt.subplots(figsize=(10, 7))
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, ax=ax)
    plt.tight_layout()
    figs['geographic_basic'] = fig3

    fig4, ax = plt.subplots(figsize=(10, 7))
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2, ax=ax)
    plt.tight_layout()
    figs['geographic_alpha'] = fig4

    fig5, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(housing["longitude"], housing["latitude"],
                         s=housing["population"] / 100,
                         c=housing["median_house_value"], cmap="jet", alpha=0.4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)
    plt.colorbar(scatter, ax=ax, label="Median House Value")
    plt.tight_layout()
    figs['geographic_value'] = fig5

    fig6 = plt.figure(figsize=(12, 8))
    attributes = ["median_house_value", "median_income", "total_rooms",
                  "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.tight_layout()
    figs['scatter_matrix'] = fig6

    fig7, ax = plt.subplots(figsize=(10, 6))
    housing.plot(kind="scatter", x="median_income", y="median_house_value",
                 alpha=0.1, grid=True, ax=ax)
    plt.tight_layout()
    figs['income_vs_value'] = fig7

    fig8, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    housing["population"].hist(ax=axs[0], bins=50)
    housing["population"].apply(np.log).hist(ax=axs[1], bins=50)
    axs[0].set_xlabel("Population")
    axs[1].set_xlabel("Log of population")
    axs[0].set_ylabel("Number of districts")
    plt.tight_layout()
    figs['population_dist'] = fig8

    fig9, ax1 = plt.subplots(figsize=(10, 6))
    ages = np.linspace(housing["housing_median_age"].min(),
                       housing["housing_median_age"].max(),
                       500).reshape(-1, 1)
    gamma1 = 0.1
    gamma2 = 0.03
    rbf1 = rbf_kernel(ages, [[35]], gamma=gamma1)
    rbf2 = rbf_kernel(ages, [[35]], gamma=gamma2)
    
    ax1.set_xlabel("Housing median age")
    ax1.set_ylabel("Number of districts")
    ax1.hist(housing["housing_median_age"], bins=50)
    
    ax2 = ax1.twinx()
    color = "blue"
    ax2.plot(ages, rbf1, color=color, label="gamma = 0.10")
    ax2.plot(ages, rbf2, color=color, label="gamma = 0.03", linestyle="--")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel("Age similarity", color=color)
    
    plt.legend(loc="upper left")
    plt.tight_layout()
    figs['age_similarity'] = fig9

    fig10, ax = plt.subplots(figsize=(10, 7))
    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
    similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]])
    
    housing_renamed = housing.rename(columns={
        "latitude": "Latitude", "longitude": "Longitude",
        "population": "Population",
        "median_house_value": "Median house value (ᴜsᴅ)"})
    housing_renamed["Max cluster similarity"] = similarities.max(axis=1)
    
    scatter = ax.scatter(housing_renamed["Longitude"], housing_renamed["Latitude"],
                         s=housing_renamed["Population"] / 100,
                         c=housing_renamed["Max cluster similarity"],
                         cmap="jet", alpha=0.4, label="Population")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)
    plt.colorbar(scatter, ax=ax, label="Max cluster similarity")
    ax.plot(cluster_simil.kmeans_.cluster_centers_[:, 1],
             cluster_simil.kmeans_.cluster_centers_[:, 0],
             linestyle="", color="black", marker="X", markersize=20,
             label="Cluster centers")
    ax.legend(loc="upper right")
    plt.tight_layout()
    figs['cluster_similarity'] = fig10

    return figs

