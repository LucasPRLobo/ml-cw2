import numpy as np
from src.preprocessing import preprocess_identity, preprocess_pca
from src.typicality import compute_typicality
from src.clustering import cluster_standard
from src.selection import select_max_typicality
from src.evaluation import evaluate_selection

# Registry of available components
PREPROCESSORS = {
    'none': preprocess_identity,
    'pca': preprocess_pca,
}

CLUSTERERS = {
    'standard': cluster_standard,
}

SELECTORS = {
    'max_typicality': select_max_typicality,
}


def run_pipeline(embeddings, labels, test_embeddings, test_labels, config):
    """
    Run one full TypiClust pipeline with given config.

    config keys:
        preprocess: str - 'none', 'pca', 'umap'
        cluster: str - 'standard', 'overclustering'
        selection: str - 'max_typicality', 'hybrid'
        budget: int
        k_typicality: int (default 20)
        random_state: int (default 42)
        + any method-specific params (pca_dims, cluster_mult, alpha, etc.)
    """
    budget = config.get('budget', 10)
    k_typ = config.get('k_typicality', 20)
    rs = config.get('random_state', 42)

    # 1. Preprocess
    preprocess_fn = PREPROCESSORS[config.get('preprocess', 'none')]

    proc_embeddings, model = preprocess_fn(embeddings, **config)
    proc_test, _ = preprocess_fn(test_embeddings, fitted_model=model)
    # 2. Compute typicality on preprocessed embeddings
    typicality = compute_typicality(proc_embeddings, k=k_typ)

    # 3. Cluster
    cluster_fn = CLUSTERERS[config.get('cluster', 'standard')]
    cluster_labels = cluster_fn(proc_embeddings, budget=budget, random_state=rs, **config)

    # 4. Select
    select_fn = SELECTORS[config.get('selection', 'max_typicality')]
    selected_indices = select_fn(
        proc_embeddings, typicality, cluster_labels, budget=budget, **config
    )

    # 5. Evaluate (use preprocessed embeddings for consistency)
    results = evaluate_selection(
        selected_indices, proc_embeddings, labels, proc_test, test_labels
    )
    results['config'] = config.copy()
    results['selected_indices'] = selected_indices.tolist()

    return results


def run_experiment_grid(embeddings, labels, test_embeddings, test_labels, configs):
    """Run pipeline for each config in the list. Returns list of result dicts."""
    results = []
    for i, config in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] {config}")
        try:
            result = run_pipeline(embeddings, labels, test_embeddings, test_labels, config)
            results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({'config': config, 'error': str(e)})
    return results
