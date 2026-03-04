import numpy as np


def select_max_typicality(embeddings, typicality, cluster_labels, budget, **kwargs):
    selected = []
    for cluster in range(budget):
        cluster_idx = np.where(cluster_labels == cluster)[0]
        best = cluster_idx[np.argmax(typicality[cluster_idx])]
        selected.append(best)
    return np.array(selected)
