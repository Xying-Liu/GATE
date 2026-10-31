# Graph-Based Clustering with Testing-Guided Cluster Number Estimation
GATE automatically determines the number of clusters by incorporating hypothesis testing into a graph-based clustering framework.

## Usage

Download this repository and extract the files. The repository contains:

- `main.py` implements GATE.
- `metrics.py` computes clustering accuracy (ACC) and the adjusted Rand index (ARI).
- `Datasets/` contains the datasets used in the experiments.

Run the main Python script to perform clustering on the provided datasets. Please refer to the source code for parameter settings and dataset configurations.

## Parameters

The following parameter settings are used in the experiments:

- `k = 9`: Number of nearest neighbors used to construct the graph.
- `m = 2`: Parameter used to identify boundary vertices.
- `alpha = 0.05`: Significance level for hypothesis testing.
- | Partitioning method (`SPLITTER`) | `girvan_newman` | Algorithm used to generate a candidate bipartition of a connected graph. |

GATE can use different graph partitioning algorithms to generate candidate splits. The implementation supports Girvan–Newman (`girvan_newman`), normalized cut (`normalized_cut`), and Kernighan–Lin (`kernighan_lin`). We use **Girvan–Newman** in all experiments.

For high-dimensional datasets, principal component analysis (PCA) is applied to retain at least 95% of the total variance.
