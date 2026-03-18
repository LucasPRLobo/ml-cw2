import numpy as np


def select_max_typicality(embeddings, typicality, cluster_labels, budget, **kwargs):
    selected = []
    for cluster in range(budget):
        cluster_idx = np.where(cluster_labels == cluster)[0]
        best = cluster_idx[np.argmax(typicality[cluster_idx])]
        selected.append(best)
    return np.array(selected)


def select_hybrid(embeddings, typicality, cluster_labels, budget, alpha=0.5, **kwargs):

    selected = []

    cluster_order = []
    for c in range(budget):
        cluster_idx = np.where(cluster_labels == c)[0]
        cluster_order.append((c, np.max(typicality[cluster_idx])))
    cluster_order.sort(key=lambda x: x[1], reverse=True)

    first_cluster = cluster_order[0][0]
    cluster_idx = np.where(cluster_labels == first_cluster)[0]
    best = cluster_idx[np.argmax(typicality[cluster_idx])]
    selected.append(best)

    for cluster in cluster_order[1:]:
        cluster_idx = np.where(cluster_labels == cluster[0])[0]
        # get canditate embeddigns
        candidate_embeddings = embeddings[cluster_idx]
        # COmpute distance to selected points
        dist = np.linalg.norm(candidate_embeddings[:, None] - embeddings[selected], axis=2)
        min_distance = dist.min(axis=1)
        
        # Normalize distancess to [0, 1]
        norm_distance = (min_distance - min_distance.min()) / (min_distance.max() - min_distance.min() + 1e-8)

        norm_typicality = (typicality[cluster_idx] - typicality[cluster_idx].min()) / (typicality[cluster_idx].max() - typicality[cluster_idx].min() + 1e-8)

        score = alpha * norm_typicality + (1 - alpha) * norm_distance

        best = cluster_idx[np.argmax(score)]
        selected.append(best)

    return np.array(selected)