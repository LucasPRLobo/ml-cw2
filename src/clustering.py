import numpy as np
from sklearn.cluster import KMeans


def cluster_standard(embeddings, budget, random_state=42, **kwargs):
    kmeans = KMeans(n_clusters=budget, random_state=random_state, n_init=10)
    kmeans.fit(embeddings)
    return kmeans.labels_
