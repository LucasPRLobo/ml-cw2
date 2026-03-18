import numpy as np
from sklearn.cluster import KMeans


def cluster_standard(embeddings, budget, random_state=42, **kwargs):
    kmeans = KMeans(n_clusters=budget, random_state=random_state, n_init=10)
    kmeans.fit(embeddings)
    return kmeans.labels_


def cluster_overclustering(embeddings, budget, cluster_mult=5, random_state=42, **kwargs):
    n_clusters = budget * cluster_mult
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(embeddings)

    points_per_cluster = np.bincount(kmeans.labels_)
    sorted_clusters = np.argsort(points_per_cluster)[-budget:]
    new_labels = np.full_like(kmeans.labels_, -1)

    for i, cluster in enumerate(sorted_clusters):
        new_labels[kmeans.labels_ == cluster] = i

    return new_labels