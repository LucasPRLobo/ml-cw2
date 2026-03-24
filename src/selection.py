import numpy as np
from sklearn.linear_model import LogisticRegression


def select_random(embeddings, budget, rng=None, **kwargs):
    """Random selection baseline."""
    if rng is None:
        rng = np.random.RandomState(42)
    return rng.choice(len(embeddings), size=budget, replace=False)


def select_uncertainty(embeddings, labels, labelled_idx, budget, **kwargs):
    """Select points with lowest max softmax probability (most uncertain)."""
    unlabelled_idx = np.setdiff1d(np.arange(len(embeddings)), labelled_idx)
    if len(labelled_idx) == 0:
        return np.random.RandomState(42).choice(len(embeddings), size=budget, replace=False)
    clf = LogisticRegression(max_iter=1000, C=100)
    clf.fit(embeddings[labelled_idx], labels[labelled_idx])
    probs = clf.predict_proba(embeddings[unlabelled_idx])
    max_probs = probs.max(axis=1)
    selected = unlabelled_idx[np.argsort(max_probs)[:budget]]
    return selected


def select_margin(embeddings, labels, labelled_idx, budget, **kwargs):
    """Select points with smallest margin between top-2 softmax probabilities."""
    unlabelled_idx = np.setdiff1d(np.arange(len(embeddings)), labelled_idx)
    if len(labelled_idx) == 0:
        return np.random.RandomState(42).choice(len(embeddings), size=budget, replace=False)
    clf = LogisticRegression(max_iter=1000, C=100)
    clf.fit(embeddings[labelled_idx], labels[labelled_idx])
    probs = clf.predict_proba(embeddings[unlabelled_idx])
    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    selected = unlabelled_idx[np.argsort(margins)[:budget]]
    return selected


def select_entropy(embeddings, labels, labelled_idx, budget, **kwargs):
    """Select points with highest softmax entropy."""
    unlabelled_idx = np.setdiff1d(np.arange(len(embeddings)), labelled_idx)
    if len(labelled_idx) == 0:
        return np.random.RandomState(42).choice(len(embeddings), size=budget, replace=False)
    clf = LogisticRegression(max_iter=1000, C=100)
    clf.fit(embeddings[labelled_idx], labels[labelled_idx])
    probs = clf.predict_proba(embeddings[unlabelled_idx])
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    selected = unlabelled_idx[np.argsort(entropy)[-budget:]]
    return selected


def select_coreset(embeddings, labelled_idx, budget, **kwargs):
    """Greedy k-center coreset selection — maximize min distance to labelled set.
    Uses incremental distance updates to avoid recomputing all pairwise distances."""
    unlabelled_idx = np.setdiff1d(np.arange(len(embeddings)), labelled_idx)
    if len(labelled_idx) == 0:
        seed = np.random.RandomState(42).choice(len(embeddings))
        selected = [seed]
        unlabelled_mask = np.ones(len(embeddings), dtype=bool)
        unlabelled_mask[seed] = False
        # Initialize min distances to the seed
        min_dists = np.linalg.norm(embeddings - embeddings[seed], axis=1)
        min_dists[seed] = -1
        remaining = budget - 1
    else:
        selected = list(labelled_idx)
        unlabelled_mask = np.ones(len(embeddings), dtype=bool)
        unlabelled_mask[labelled_idx] = False
        # Initialize min distances to all labelled points
        min_dists = np.full(len(embeddings), np.inf)
        for idx in labelled_idx:
            d = np.linalg.norm(embeddings - embeddings[idx], axis=1)
            min_dists = np.minimum(min_dists, d)
        min_dists[labelled_idx] = -1
        remaining = budget

    for _ in range(remaining):
        # Select the unlabelled point with largest min distance
        candidates = np.where(unlabelled_mask)[0]
        best_pos = candidates[np.argmax(min_dists[candidates])]
        selected.append(best_pos)
        unlabelled_mask[best_pos] = False
        # Update min distances with the newly selected point
        new_dists = np.linalg.norm(embeddings - embeddings[best_pos], axis=1)
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[best_pos] = -1

    if len(labelled_idx) == 0:
        return np.array(selected)
    return np.array(selected[len(labelled_idx):])


def _mc_dropout_predict(embeddings, labels, labelled_idx, unlabelled_idx,
                        n_forward=20, n_classes=10, hidden=128, n_epochs=100, dropout=0.5):
    """Train a small network with dropout on labelled data,
    then run MC dropout inference on unlabelled data."""
    import torch
    import torch.nn as nn

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = embeddings.shape[1]

    # Small MLP with dropout
    model = nn.Sequential(
        nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, n_classes)
    ).to(device)

    # Train
    X_train = torch.FloatTensor(embeddings[labelled_idx]).to(device)
    y_train = torch.LongTensor(labels[labelled_idx]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

    # MC dropout inference (keep dropout ON)
    model.train()  # keeps dropout active
    X_test = torch.FloatTensor(embeddings[unlabelled_idx]).to(device)
    all_probs = []
    with torch.no_grad():
        for _ in range(n_forward):
            logits = model(X_test)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    return np.stack(all_probs)  # (n_forward, n_unlabelled, n_classes)


def select_bald(embeddings, labels, labelled_idx, budget, **kwargs):
    """BALD: Bayesian Active Learning by Disagreement.
    Select points with highest mutual information between predictions and model parameters.
    MI = H[y|x] - E[H[y|x,w]] (entropy of mean - mean of entropies)."""
    unlabelled_idx = np.setdiff1d(np.arange(len(embeddings)), labelled_idx)
    if len(labelled_idx) == 0:
        return np.random.RandomState(42).choice(len(embeddings), size=budget, replace=False)

    all_probs = _mc_dropout_predict(embeddings, labels, labelled_idx, unlabelled_idx)
    # Mean prediction across forward passes
    mean_probs = all_probs.mean(axis=0)
    # Entropy of mean prediction
    H_mean = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
    # Mean entropy across forward passes
    entropies = -np.sum(all_probs * np.log(all_probs + 1e-10), axis=2)
    mean_H = entropies.mean(axis=0)
    # BALD score = mutual information
    bald_scores = H_mean - mean_H
    selected = unlabelled_idx[np.argsort(bald_scores)[-budget:]]
    return selected


def select_dbal(embeddings, labels, labelled_idx, budget, **kwargs):
    """DBAL: Deep Bayesian Active Learning.
    Select points with highest average predictive entropy under MC dropout."""
    unlabelled_idx = np.setdiff1d(np.arange(len(embeddings)), labelled_idx)
    if len(labelled_idx) == 0:
        return np.random.RandomState(42).choice(len(embeddings), size=budget, replace=False)

    all_probs = _mc_dropout_predict(embeddings, labels, labelled_idx, unlabelled_idx)
    # Average entropy across MC samples
    entropies = -np.sum(all_probs * np.log(all_probs + 1e-10), axis=2)
    mean_entropy = entropies.mean(axis=0)
    selected = unlabelled_idx[np.argsort(mean_entropy)[-budget:]]
    return selected


def select_badge(embeddings, labels, labelled_idx, budget, **kwargs):
    """BADGE: Batch Active learning by Diverse Gradient Embeddings.
    Uses gradient embeddings from the last layer of a trained linear model,
    then applies k-means++ initialization to select diverse uncertain points."""
    unlabelled_idx = np.setdiff1d(np.arange(len(embeddings)), labelled_idx)
    if len(labelled_idx) == 0:
        return np.random.RandomState(42).choice(len(embeddings), size=budget, replace=False)

    clf = LogisticRegression(max_iter=1000, C=100)
    clf.fit(embeddings[labelled_idx], labels[labelled_idx])

    # Compute gradient embeddings for unlabelled points
    probs = clf.predict_proba(embeddings[unlabelled_idx])
    predicted = probs.argmax(axis=1)
    n_classes = probs.shape[1]

    # Gradient embedding: for each point, the gradient of the loss w.r.t.
    # the last layer weights is (p - y) ⊗ x, where y is one-hot predicted label
    # We use the predicted label's gradient: (1 - p_hat) * x for the predicted class
    grad_embeddings = []
    for i in range(len(unlabelled_idx)):
        p = probs[i]
        x = embeddings[unlabelled_idx[i]]
        # Gradient for predicted class
        one_hot = np.zeros(n_classes)
        one_hot[predicted[i]] = 1.0
        grad = np.outer(p - one_hot, x).flatten()
        grad_embeddings.append(grad)
    grad_embeddings = np.array(grad_embeddings)

    # k-means++ initialization on gradient embeddings to select diverse uncertain points
    selected_local = []
    # First point: highest gradient norm (most uncertain)
    norms = np.linalg.norm(grad_embeddings, axis=1)
    first = np.argmax(norms)
    selected_local.append(first)

    for _ in range(budget - 1):
        # Distance to nearest selected point
        dists = np.linalg.norm(
            grad_embeddings[:, None] - grad_embeddings[selected_local][None, :],
            axis=2
        )
        min_dists = dists.min(axis=1)
        min_dists[selected_local] = 0
        # Proportional to distance squared
        probs_select = min_dists ** 2
        probs_select /= probs_select.sum() + 1e-10
        next_idx = np.random.RandomState(42).choice(len(grad_embeddings), p=probs_select)
        selected_local.append(next_idx)

    return unlabelled_idx[np.array(selected_local)]


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
        candidate_embeddings = embeddings[cluster_idx]
        dist = np.linalg.norm(candidate_embeddings[:, None] - embeddings[selected], axis=2)
        min_distance = dist.min(axis=1)

        norm_distance = (min_distance - min_distance.min()) / (min_distance.max() - min_distance.min() + 1e-8)
        norm_typicality = (typicality[cluster_idx] - typicality[cluster_idx].min()) / (typicality[cluster_idx].max() - typicality[cluster_idx].min() + 1e-8)

        score = alpha * norm_typicality + (1 - alpha) * norm_distance

        best = cluster_idx[np.argmax(score)]
        selected.append(best)

    return np.array(selected)
