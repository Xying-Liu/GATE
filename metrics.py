import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


def clustering_accuracy(y_true, y_pred):
    """用匈牙利算法最优匹配标签，支持真实类别数与预测簇数不同。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("标签必须是一维数组")
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("真实标签和预测标签必须等长且非空")

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
