import numpy as np
from sklearn.decomposition import PCA


def preprocess_identity(embeddings, **kwargs):
    return embeddings, None


def preprocess_pca(embeddings, n_components=100, fitted_model=None, **kwargs):
    if fitted_model is None:
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(embeddings)
        return transformed, pca
    else:
        return fitted_model.transform(embeddings), fitted_model
