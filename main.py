from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import hypergeom
from sklearn.cluster import SpectralClustering
from sklearn.datasets import load_iris, load_wine, make_blobs, make_circles
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from metrics import evaluate

DATA_DIR = Path(__file__).resolve().parent / "Datasets"
SINGLE_DATASET = "iris"
DELIMITER = ","

GRAPH_K = 9
SPLITTER = "girvan_newman"  # girvan_newman | normalized_cut | kernighan_lin
MNN_K = 2                 # Number of nearest neighbors considered on each side of a candidate split.
ALPHA = 0.05
MIN_CLUSTER_SIZE = 10
RANDOM_STATE = None
RUN_ALL_DATASETS = True
VISUALIZE_DATASET = None  # Show recursive splits for this dataset. Set to None to disable plots.
USE_PCA_FOR_HIGH_DIM = True
PCA_MIN_FEATURES = 3           # Apply PCA when the original feature count reaches this threshold.
PCA_VARIANCE = 0.95          # Target cumulative explained variance.


def load_numeric_dataset(path, label_col=-1):
    if not path.is_file():
        # Locate the requested file in the repository's dataset subdirectories.
        matches = sorted(p for p in DATA_DIR.rglob("*")
                         if p.is_file() and p.name.casefold() == path.name.casefold())
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one file named {path.name} under {DATA_DIR}. Found: {matches}")
        path = matches[0]
    data = np.genfromtxt(path, delimiter=DELIMITER, dtype=float,
                         missing_values="?", filling_values=np.nan)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    y = data[:, label_col]
    X = np.delete(data, label_col, axis=1)
    keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[keep], y[keep]


def gen_cos_sin(n=68):
    x = np.linspace(-2 * np.pi, 2 * np.pi, n // 2)
    X = np.vstack([np.c_[x, np.sin(x)], np.c_[x, np.cos(x)]])
    return X, np.repeat([0, 1], len(x))


def gen_circles(n=2, n_samples=75):
    del n  # Keep the original call signature. This generator produces two rings.
    return make_circles(n_samples=2 * n_samples, factor=.45, noise=.04,
                        random_state=RANDOM_STATE)


def get_blobs(n_samples=150):
    return make_blobs(n_samples=n_samples, centers=3, cluster_std=.65,
                      random_state=RANDOM_STATE)


def get_iris():
    data = load_iris()
    return data.data, data.target


def get_wine():
    data = load_wine()
    return data.data, data.target


DATASET_LOADERS = [
    ("aggregation", lambda: load_numeric_dataset(DATA_DIR / "aggregation.txt")),
    ("blobs", lambda: get_blobs(n_samples=150)),
    ("circles", lambda: gen_circles(n=2, n_samples=75)),
    ("flame", lambda: load_numeric_dataset(DATA_DIR / "flame.txt")),
    ("iusn", lambda: load_numeric_dataset(DATA_DIR / "iusn.txt")),
    ("jain", lambda: load_numeric_dataset(DATA_DIR / "jain.txt")),
    ("pathbased", lambda: load_numeric_dataset(DATA_DIR / "pathbased.txt")),
    ("R15", lambda: load_numeric_dataset(DATA_DIR / "R15.txt")),
    ("cos_sin", gen_cos_sin),
    ("spiral", lambda: load_numeric_dataset(DATA_DIR / "spiral.txt")),
    ("zelink1", lambda: load_numeric_dataset(DATA_DIR / "zelink1.txt")),
    ("zelink5", lambda: load_numeric_dataset(DATA_DIR / "zelink5.txt")),


    ("Auto", lambda: load_numeric_dataset(DATA_DIR / "Auto+MPG.txt")),
    ("exasens", lambda: load_numeric_dataset(DATA_DIR / "exasens.csv")),
    ("glass", lambda: load_numeric_dataset(DATA_DIR / "glass.txt")),
    ("haberman", lambda: load_numeric_dataset(DATA_DIR / "haberman.data")),
    ("hcv", lambda: load_numeric_dataset(DATA_DIR / "hcv_0.txt", label_col=0)),
    ("iris", get_iris),
    ("new_thyroid_0", lambda: load_numeric_dataset(DATA_DIR / "new_thyroid_0.data", label_col=0)),
    ("parkinsons2", lambda: load_numeric_dataset(DATA_DIR / "outlier/parkinsons2.txt")),
    ("seeds", lambda: load_numeric_dataset(DATA_DIR / "seeds.txt")),
    ("toxicity", lambda: load_numeric_dataset(DATA_DIR / "toxicity.txt")),
    ("user", lambda: load_numeric_dataset(DATA_DIR / "user.txt")),
    ("wine", get_wine),

]


def build_fixed_knn(X):
    k = min(GRAPH_K, len(X) - 1)
    A = kneighbors_graph(X, k,mode="connectivity", include_self=False)
    A = ((A + A.T) > 0).astype(np.uint8)
    return nx.from_scipy_sparse_array(A)


def prepare_features(X_raw):
    """Min-max scale the features and apply PCA when the dimension threshold is met."""
    X_scaled = MinMaxScaler().fit_transform(np.asarray(X_raw, dtype=float))
    original_dim = X_scaled.shape[1]
    if not USE_PCA_FOR_HIGH_DIM or original_dim < PCA_MIN_FEATURES:
        return X_scaled, original_dim, original_dim, 1.0

    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    X_reduced = pca.fit_transform(X_scaled)
    explained = float(pca.explained_variance_ratio_.sum())
    return X_reduced, original_dim, X_reduced.shape[1], explained

def split_graph(G):
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if len(components) > 1:
        return set(components[0]), set().union(*components[1:]), "components"

    if SPLITTER == "girvan_newman":
        parts = list(next(nx.community.girvan_newman(G)))
        parts.sort(key=len, reverse=True)
        return set(parts[0]), set().union(*parts[1:]), SPLITTER

    if SPLITTER == "kernighan_lin":
        a, b = nx.community.kernighan_lin_bisection(G, seed=RANDOM_STATE)
        return set(a), set(b), SPLITTER

    if SPLITTER == "normalized_cut":
        nodes = np.asarray(list(G.nodes()), dtype=int)
        A = nx.to_numpy_array(G, nodelist=nodes)
        labels = SpectralClustering(
            n_clusters=2, affinity="precomputed", assign_labels="kmeans",
            n_init=20, random_state=RANDOM_STATE,
        ).fit_predict(A)
        return set(nodes[labels == 0]), set(nodes[labels == 1]), SPLITTER

    raise ValueError(f"Unknown SPLITTER: {SPLITTER}")


def mutual_cross_neighbors(X, left, right, k):
    """Identify mutual cross-group neighbors using distances in feature space."""
    left = np.asarray(sorted(left), dtype=int)
    right = np.asarray(sorted(right), dtype=int)
    D = cdist(X[left], X[right])
    k = min(k, len(left), len(right))
    right_knn = np.argpartition(D, k - 1, axis=1)[:, :k]
    left_knn = np.argpartition(D, k - 1, axis=0)[:k, :]
    pairs = []
    for i in range(len(left)):
        for j in right_knn[i]:
            if i in left_knn[:, j]:
                pairs.append((int(left[i]), int(right[j])))
    return pairs, {u for u, _ in pairs}, {v for _, v in pairs}


def boundary_hypergeom_test(G, boundary_left, boundary_right):
    """Under the null, all sets of M edges among the possible vertex pairs are equally likely."""
    B = boundary_left | boundary_right
    G_B = G.subgraph(B).copy()
    n1, n2 = len(boundary_left), len(boundary_right)
    N = n1 + n2
    L = N * (N - 1) // 2
    K_cross = n1 * n2
    M = G_B.number_of_edges()
    cross_edges = [(u, v) for u, v in G_B.edges()
                   if (u in boundary_left) != (v in boundary_left)]
    c_obs = len(cross_edges)
    if N < 4 or n1 == 0 or n2 == 0 or M == 0:
        return 1.0, 0.0, c_obs, M, G_B, cross_edges
    expected = M * K_cross / L
    p_value = float(hypergeom.cdf(c_obs, L, K_cross, M))
    return p_value, expected, c_obs, M, G_B, cross_edges


def plot_step(X, G, nodes, left, right, B_left, B_right, G_B, cross_edges,
              path, p_value, expected, c_obs, M, decision):
    Z = X[:, :2] if X.shape[1] >= 2 else np.c_[X[:, 0], np.zeros(len(X))]
    pos = {i: Z[i] for i in nodes}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    nx.draw_networkx_edges(G, pos, ax=axes[0], edge_color="0.82", width=.6)
    axes[0].scatter(Z[list(left), 0], Z[list(left), 1], s=22, c="#2878B5",
                    label=f"left={len(left)}")
    axes[0].scatter(Z[list(right), 0], Z[list(right), 1], s=22, c="#D95319",
                    label=f"right={len(right)}")
    axes[0].scatter(Z[list(B_left), 0], Z[list(B_left), 1], s=85,
                    facecolors="none", edgecolors="blue", linewidths=1.5)
    axes[0].scatter(Z[list(B_right), 0], Z[list(B_right), 1], s=85,
                    facecolors="none", edgecolors="red", linewidths=1.5)
    axes[0].set_title(f"{path}: candidate split and MNN boundary")
    axes[0].legend()

    B = list(B_left | B_right)
    if B:
        nx.draw_networkx_edges(G_B, pos, ax=axes[1], edge_color="0.65", width=1)
        nx.draw_networkx_edges(G_B.edge_subgraph(cross_edges), pos, ax=axes[1],
                               edge_color="red", width=2.2)
        axes[1].scatter(Z[list(B_left), 0], Z[list(B_left), 1], s=45, c="#2878B5")
        axes[1].scatter(Z[list(B_right), 0], Z[list(B_right), 1], s=45, c="#D95319")
    axes[1].set_title(
        f"Boundary induced graph: M={M}, cross={c_obs}, E0={expected:.2f}\n"
        f"left-tail p={p_value:.4g}: {decision}"
    )
    for ax in axes:
        ax.axis("off")
        ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    plt.show()


def recursive_clustering(X, G_global, visualize=False):
    leaves, history = [], []

    def visit(nodes, path):
        nodes = np.asarray(sorted(nodes), dtype=int)
        G = G_global.subgraph(nodes).copy()  # Induced subgraph of the fixed global graph.
        if len(nodes) < MIN_CLUSTER_SIZE:
            leaves.append(nodes)
            history.append((path, len(nodes), 0, 0, 0, 0, 1.0, "STOP_TOO_SMALL"))
            return

        left, right, method = split_graph(G)
        if min(len(left), len(right)) < MIN_CLUSTER_SIZE:
            leaves.append(nodes)
            history.append((path, len(nodes), len(left), len(right), 0, 0, 1.0,
                            "STOP_SMALL_CHILD"))
            return

        # Accept disconnected splits after the size check. Test connected graphs at the boundary.
        if method == "components":
            history.append((path, len(nodes), len(left), len(right), 0, 0, 0.0,
                            "SPLIT_DISCONNECTED"))
            visit(left, path + "L")
            visit(right, path + "R")
            return

        _, B_left, B_right = mutual_cross_neighbors(X, left, right, MNN_K)
        p, expected, c_obs, M, G_B, cross_edges = boundary_hypergeom_test(
            G, B_left, B_right
        )
        decision = "SPLIT" if p <= ALPHA else "STOP"
        history.append((path, len(nodes), len(left), len(right), c_obs, M, p, decision))
        if visualize:
            plot_step(X, G, nodes, left, right, B_left, B_right, G_B,
                      cross_edges, path, p, expected, c_obs, M, decision)
        if decision == "SPLIT":
            visit(left, path + "L")
            visit(right, path + "R")
        else:
            leaves.append(nodes)

    visit(np.arange(len(X)), "root")
    labels = np.full(len(X), -1, dtype=int)
    for label, nodes in enumerate(leaves):
        labels[nodes] = label
    return labels, history

def run_dataset(name, loader):
    X_raw, y_true = loader()
    X, _, _, _ = prepare_features(X_raw)
    labels, history = recursive_clustering(
        X, build_fixed_knn(X), visualize=(name == VISUALIZE_DATASET)
    )
    print(f"{name}: estimated K = {len(np.unique(labels))}, "
          f"cluster sizes = {np.bincount(labels).tolist()}")
    acc, ari = evaluate(y_true, labels)
    print(f"ACC = {acc:.4f}, ARI = {ari:.4f}")
    return labels, history


def main():
    datasets = dict(DATASET_LOADERS)
    names = datasets if RUN_ALL_DATASETS else [SINGLE_DATASET]
    for name in names:
        run_dataset(name, datasets[name])


if __name__ == "__main__":
    main()
