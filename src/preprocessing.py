import numpy as np
from sklearn.decomposition import PCA
import umap

def preprocess_identity(embeddings, **kwargs):
    return embeddings, None


def preprocess_pca(embeddings, n_components=100, fitted_model=None, **kwargs):
    if fitted_model is None:
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(embeddings)
        return transformed, pca
    else:
        return fitted_model.transform(embeddings), fitted_model

def preprocess_umap(embeddings, n_components=2, fitted_model=None, random_state=42, **kwargs):
    if fitted_model is None:
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        transformed = reducer.fit_transform(embeddings)
        return transformed, reducer
    else:
        return fitted_model.transform(embeddings), fitted_model