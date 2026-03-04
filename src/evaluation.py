import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def linear_probe(train_embeddings, train_labels, test_embeddings, test_labels):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_embeddings, train_labels)
    preds = clf.predict(test_embeddings)
    return accuracy_score(test_labels, preds)


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
