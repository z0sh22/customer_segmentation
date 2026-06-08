import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator

def find_opt(X_scaled, kr = range(1, 11)) -> list:
    wcss = []

    for k in kr:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
    
    kl = KneeLocator(kr, wcss, curve='convex', direction='decreasing')
    opt_k = kl.knee
    return wcss, opt_k

def train_kmeans(X_scaled, n_clusters: int):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, labels