import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

import torchvision.models as models

from src.selection import (
    select_random, select_uncertainty, select_margin,
    select_entropy, select_coreset, select_badge,
    select_bald, select_dbal, select_max_typicality
)


# ============================================================
# Framework 1: Fully Supervised (train ResNet-18 from scratch)
# ============================================================

def _make_resnet18_cifar(num_classes=10):
    """ResNet-18 adapted for CIFAR-10 (32x32 images)."""
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def train_supervised(train_images, train_labels, test_images, test_labels,
                     n_epochs=200, lr=0.025, device='cuda'):
    """Train ResNet-18 from scratch on labelled images and evaluate.
    Follows paper's F.2.1: SGD with 0.9 momentum (Nesterov), cosine LR,
    random crops + horizontal flips, weight re-init each call."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    # Re-initialize weights each time (paper: prevents overconfident predictions)
    model = _make_resnet18_cifar().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    # Data augmentation: random crops + horizontal flips (as per paper)
    train_tensor = torch.FloatTensor(train_images)
    train_labels_tensor = torch.LongTensor(train_labels)
    train_ds = TensorDataset(train_tensor, train_labels_tensor)
    train_loader = DataLoader(train_ds, batch_size=min(len(train_images), 64),
                              shuffle=True, drop_last=False)

    # Training with per-image augmentation
    model.train()
    for epoch in range(n_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Per-image random horizontal flip
            flip_mask = torch.rand(x.size(0)) > 0.5
            x[flip_mask] = torch.flip(x[flip_mask], dims=[3])
            # Per-image random crop (pad 4, crop 32)
            x = nn.functional.pad(x, (4, 4, 4, 4), mode='reflect')
            augmented = torch.zeros(x.size(0), x.size(1), 32, 32, device=device)
            for img_i in range(x.size(0)):
                ci = torch.randint(0, 8, (1,)).item()
                cj = torch.randint(0, 8, (1,)).item()
                augmented[img_i] = x[img_i, :, ci:ci+32, cj:cj+32]
            optimizer.zero_grad()
            loss = criterion(model(augmented), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluation
    model.eval()
    test_ds = TensorDataset(
        torch.FloatTensor(test_images), torch.LongTensor(test_labels))
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)

    return correct / total


# ============================================================
# Framework 2: SS Embedding (linear probe on frozen embeddings)
# ============================================================

def linear_probe(train_embeddings, train_labels, test_embeddings, test_labels):
    clf = LogisticRegression(max_iter=1000, C=100)
    clf.fit(train_embeddings, train_labels)
    preds = clf.predict(test_embeddings)
    return accuracy_score(test_labels, preds)


# ============================================================
# Framework 3: Semi-Supervised (Pseudo-labelling on embeddings)
# ============================================================
# Paper uses FlexMatch (WideResNet-28, 400k iterations) which is
# computationally infeasible. We approximate with iterative
# pseudo-labelling on frozen SimCLR embeddings with per-class
# adaptive thresholds inspired by FlexMatch's curriculum.

def semi_supervised_eval(labelled_idx, labels,
                         train_images, test_images, test_labels,
                         embeddings=None, test_embeddings=None,
                         n_iterations=5, initial_threshold=0.95,
                         max_pseudo_per_iter=5000):
    """
    Semi-supervised evaluation using iterative pseudo-labelling
    on frozen SimCLR embeddings with per-class adaptive thresholds.
    """
    train_emb = embeddings[labelled_idx]
    train_lab = labels[labelled_idx].copy()
    labelled_emb = train_emb.copy()
    labelled_lab = train_lab.copy()
    n_classes = 10

    unlabelled_mask = np.ones(len(embeddings), dtype=bool)
    unlabelled_mask[labelled_idx] = False

    class_thresholds = np.full(n_classes, initial_threshold)

    for iteration in range(n_iterations):
        clf = LogisticRegression(max_iter=500, C=100)
        clf.fit(labelled_emb, labelled_lab)

        unlabelled_idx = np.where(unlabelled_mask)[0]
        if len(unlabelled_idx) == 0:
            break

        probs = clf.predict_proba(embeddings[unlabelled_idx])
        max_probs = probs.max(axis=1)
        predicted = probs.argmax(axis=1)

        confident = np.zeros(len(unlabelled_idx), dtype=bool)
        for c in range(n_classes):
            class_mask = predicted == c
            if class_mask.sum() > 0:
                confident[class_mask] = max_probs[class_mask] >= class_thresholds[c]

        if confident.sum() == 0:
            class_thresholds *= 0.95
            continue

        confident_positions = np.where(confident)[0]
        if len(confident_positions) > max_pseudo_per_iter:
            top = np.argsort(max_probs[confident_positions])[-max_pseudo_per_iter:]
            confident_positions = confident_positions[top]

        pseudo_idx = unlabelled_idx[confident_positions]
        pseudo_labels = predicted[confident_positions]

        labelled_emb = np.vstack([labelled_emb, embeddings[pseudo_idx]])
        labelled_lab = np.concatenate([labelled_lab, pseudo_labels])
        unlabelled_mask[pseudo_idx] = False

        class_counts = np.bincount(labelled_lab.astype(int), minlength=n_classes)
        max_count = class_counts.max()
        for c in range(n_classes):
            ratio = class_counts[c] / (max_count + 1e-8)
            class_thresholds[c] = initial_threshold * ratio + (1 - ratio) * 0.5

    clf = LogisticRegression(max_iter=1000, C=100)
    clf.fit(labelled_emb, labelled_lab)
    preds = clf.predict(test_embeddings)
    return accuracy_score(test_labels, preds)


# ============================================================
# Evaluation helpers
# ============================================================

def evaluate_selection(selected_indices, embeddings, labels, test_embeddings, test_labels):
    sel_emb = embeddings[selected_indices]
    sel_lab = labels[selected_indices]
    accuracy = linear_probe(sel_emb, sel_lab, test_embeddings, test_labels)
    n_classes = len(np.unique(sel_lab))
    class_counts = np.bincount(sel_lab, minlength=10)
    tv_distance = 0.5 * np.sum(np.abs(class_counts / len(sel_lab) - 1.0 / 10))
    return {
        'accuracy': accuracy,
        'n_classes_covered': n_classes,
        'tv_distance': tv_distance,
        'selected_labels': sel_lab.tolist(),
    }


def random_baseline(embeddings, labels, test_embeddings, test_labels, budget, n_seeds=30):
    accs = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(embeddings), size=budget, replace=False)
        acc = linear_probe(embeddings[idx], labels[idx], test_embeddings, test_labels)
        accs.append(acc)
    return {'mean': np.mean(accs), 'std': np.std(accs), 'all': accs}


def _compute_typicality(embeddings, k=20):
    """Compute typicality as inverse mean distance to k nearest neighbours."""
    nn_model = NearestNeighbors(n_neighbors=k + 1)
    nn_model.fit(embeddings)
    distances, _ = nn_model.kneighbors(embeddings)
    distances = distances[:, 1:]
    return np.mean(distances, axis=1) ** -1


# ============================================================
# Multi-round AL loop (supports all 3 frameworks)
# ============================================================

def run_al_rounds(embeddings, labels, test_embeddings, test_labels,
                  strategy, budget_per_round=10, n_rounds=5, n_reps=10,
                  random_state=42, framework='ss_embedding',
                  train_images=None, test_images=None,
                  all_embeddings=None, all_labels=None):
    """
    Run multi-round active learning experiment.

    Args:
        strategy: 'random', 'uncertainty', 'margin', 'entropy', 'coreset', 'typiclust'
        framework: 'fully_supervised', 'ss_embedding', 'semi_supervised'
        train_images: raw images (needed for fully_supervised)
        test_images: raw test images (needed for fully_supervised)
        all_embeddings: full training pool embeddings (needed for semi_supervised)
        all_labels: full training pool labels (needed for semi_supervised)
    """
    budgets = [(i + 1) * budget_per_round for i in range(n_rounds)]
    all_accs = np.zeros((n_reps, n_rounds))

    for rep in range(n_reps):
        seed = random_state + rep
        rng = np.random.RandomState(seed)
        labelled_idx = np.array([], dtype=int)

        for round_i in range(n_rounds):
            # --- Selection ---
            if strategy == 'random':
                unlabelled = np.setdiff1d(np.arange(len(embeddings)), labelled_idx)
                new_idx = rng.choice(unlabelled, size=budget_per_round, replace=False)

            elif strategy == 'uncertainty':
                new_idx = select_uncertainty(
                    embeddings, labels, labelled_idx, budget_per_round)

            elif strategy == 'margin':
                new_idx = select_margin(
                    embeddings, labels, labelled_idx, budget_per_round)

            elif strategy == 'entropy':
                new_idx = select_entropy(
                    embeddings, labels, labelled_idx, budget_per_round)

            elif strategy == 'coreset':
                new_idx = select_coreset(
                    embeddings, labelled_idx, budget_per_round)

            elif strategy == 'badge':
                new_idx = select_badge(
                    embeddings, labels, labelled_idx, budget_per_round)

            elif strategy == 'bald':
                new_idx = select_bald(
                    embeddings, labels, labelled_idx, budget_per_round)

            elif strategy == 'dbal':
                new_idx = select_dbal(
                    embeddings, labels, labelled_idx, budget_per_round)

            elif strategy == 'typiclust':
                # Paper: K = min(|L_{i-1}| + B, max_clusters)
                # max_clusters = 500 for CIFAR-10
                max_clusters = 500
                unlabelled_idx = np.setdiff1d(
                    np.arange(len(embeddings)), labelled_idx)
                unlabelled_emb = embeddings[unlabelled_idx]

                n_clusters = min(len(labelled_idx) + budget_per_round, max_clusters)
                kmeans = KMeans(
                    n_clusters=n_clusters, random_state=seed, n_init=10)
                kmeans.fit(unlabelled_emb)

                # Find uncovered clusters
                all_cluster_labels = kmeans.predict(embeddings)
                covered_clusters = set()
                if len(labelled_idx) > 0:
                    covered_clusters = set(all_cluster_labels[labelled_idx])

                uncovered = [c for c in range(n_clusters)
                             if c not in covered_clusters]

                # Paper: drop clusters with < 5 samples, use min(20, cluster_size) for KNN
                new_idx_local = []
                for c in uncovered:
                    if len(new_idx_local) >= budget_per_round:
                        break
                    mask = kmeans.labels_ == c
                    cluster_idx = np.where(mask)[0]
                    cluster_size = len(cluster_idx)
                    if cluster_size < 5:
                        continue  # Paper: skip small clusters
                    # Paper: use min(20, cluster_size) neighbours
                    k_typ = min(20, cluster_size - 1)
                    if k_typ < 1:
                        continue
                    typicality = _compute_typicality(
                        unlabelled_emb[cluster_idx], k=k_typ)
                    best = cluster_idx[np.argmax(typicality)]
                    new_idx_local.append(best)

                # If not enough, fill from remaining uncovered (relaxed size filter)
                if len(new_idx_local) < budget_per_round:
                    for c in uncovered:
                        if len(new_idx_local) >= budget_per_round:
                            break
                        mask = kmeans.labels_ == c
                        cluster_idx = np.where(mask)[0]
                        if any(cluster_idx[0] == idx for idx in new_idx_local):
                            continue
                        # Pick most central point by distance to centroid
                        centroid = unlabelled_emb[cluster_idx].mean(axis=0)
                        dists = np.linalg.norm(unlabelled_emb[cluster_idx] - centroid, axis=1)
                        best = cluster_idx[np.argmin(dists)]
                        new_idx_local.append(best)

                new_idx = unlabelled_idx[new_idx_local[:budget_per_round]]

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            labelled_idx = np.concatenate([labelled_idx, new_idx]).astype(int)

            # --- Evaluation (framework-dependent) ---
            if framework == 'ss_embedding':
                acc = linear_probe(
                    embeddings[labelled_idx], labels[labelled_idx],
                    test_embeddings, test_labels)

            elif framework == 'fully_supervised':
                if train_images is None or test_images is None:
                    raise ValueError("fully_supervised requires train_images and test_images")
                acc = train_supervised(
                    train_images[labelled_idx], labels[labelled_idx],
                    test_images, test_labels)

            elif framework == 'semi_supervised':
                acc = semi_supervised_eval(
                    labelled_idx, labels,
                    train_images=None, test_images=None, test_labels=test_labels,
                    embeddings=embeddings, test_embeddings=test_embeddings)

            else:
                raise ValueError(f"Unknown framework: {framework}")

            all_accs[rep, round_i] = acc

    return {
        'budgets': budgets,
        'mean': all_accs.mean(axis=0),
        'std': all_accs.std(axis=0),
        'all': all_accs,
    }
