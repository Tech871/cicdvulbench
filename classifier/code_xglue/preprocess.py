import numpy as np


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""

    max_length = max(map(len, features))

    for i in range(len(features)):
        feature = np.array(features[i])
        pad = max_length - feature.shape[0]
        feature = np.pad(feature, ((0, pad), (0, 0)), mode='constant')
        features[i] = feature

    return np.array(list(features))


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""

    adj_sum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(adj_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""

    max_length = max([a.shape[0] for a in adj])
    mask = np.zeros((len(adj), max_length, 1))

    for i, a in enumerate(adj):
        mask[i, :a.shape[0], :] = 1.
        a = normalize_adj(a)
        pad = max_length - a.shape[0]
        adj[i] = np.pad(a, ((0, pad), (0, pad)), mode='constant')

    # return coo_to_tuple(sparse.COO(np.array(list(adj)))), mask
    return np.array(list(adj)), mask
