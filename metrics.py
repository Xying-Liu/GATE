import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


def clustering_accuracy(y_true, y_pred):
    """Compute accuracy after optimal one-to-one label matching."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Labels must be one-dimensional.")
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("Label arrays must have equal, nonzero length.")

    _, true_ids = np.unique(y_true, return_inverse=True)
    _, pred_ids = np.unique(y_pred, return_inverse=True)

    counts = np.zeros(
        (true_ids.max() + 1, pred_ids.max() + 1),
        dtype=np.int64,
    )
    np.add.at(counts, (true_ids, pred_ids), 1)
    rows, cols = linear_sum_assignment(-counts)
    return float(counts[rows, cols].sum() / len(y_true))


def evaluate(y_true, y_pred):
    return clustering_accuracy(y_true, y_pred), adjusted_rand_score(y_true, y_pred)
