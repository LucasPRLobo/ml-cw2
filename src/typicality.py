import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_typicality(embeddings, k=20):
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    distances = distances[:, 1:]  # exclude self
    return np.mean(distances, axis=1) ** -1
