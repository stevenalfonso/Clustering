import numpy as np

def dbscan(X, eps, min_samples):
    """
    Perform DBSCAN clustering algorithm.

    Args:
    - X: A numpy array of shape (n_samples, n_features) representing the dataset to cluster.
    - eps: A float representing the maximum distance between two samples for them to be considered as in the same neighborhood.
    - min_samples: An int representing the minimum number of samples required to form a dense region.

    Returns:
    - labels: A numpy array of shape (n_samples,) representing the cluster labels assigned to each sample.
             Noise points are labeled as -1.
    """

    # Initialize variables
    n_samples = X.shape[0]
    visited = np.zeros(n_samples, dtype=bool)
    labels = np.zeros(n_samples, dtype=int)
    cluster_id = 0

    # Iterate over each sample in the dataset
    for i in range(n_samples):
        if not visited[i]:
            visited[i] = True

            # Find all samples within eps distance of the current sample
            neighbors = find_neighbors(X, i, eps)

            # If the number of neighbors is less than min_samples, mark the sample as noise
            if len(neighbors) < min_samples:
                labels[i] = -1
            else:
                # Expand the current cluster
                cluster_id += 1
                labels[i] = cluster_id
                expand_cluster(X, visited, labels, i, neighbors, cluster_id, eps, min_samples)

    return labels


def find_neighbors(X, i, eps):
    """
    Find all samples within eps distance of the i-th sample in the dataset.

    Args:
    - X: A numpy array of shape (n_samples, n_features) representing the dataset.
    - i: An int representing the index of the sample to find neighbors for.
    - eps: A float representing the maximum distance between two samples for them to be considered as in the same neighborhood.

    Returns:
    - neighbors: A list of indices representing the neighbors of the i-th sample in the dataset.
    """
    neighbors = []
    for j in range(X.shape[0]):
        if np.linalg.norm(X[i] - X[j]) < eps:
            neighbors.append(j)
    return neighbors


def expand_cluster(X, visited, labels, i, neighbors, cluster_id, eps, min_samples):
    """
    Expand the current cluster by adding all reachable samples to it.

    Args:
    - X: A numpy array of shape (n_samples, n_features) representing the dataset.
    - visited: A boolean numpy array of shape (n_samples,) representing which samples have already been visited.
    - labels: A numpy array of shape (n_samples,) representing the cluster labels assigned to each sample.
    - i: An int representing the index of the sample to start expanding from.
    - neighbors: A list of indices representing the neighbors of the i-th sample in the dataset.
    - cluster_id: An int representing the cluster ID to assign to all samples in the current cluster.
    - eps: A float representing the maximum distance between two samples for them to be considered as in the same neighborhood.
    - min_samples: An int representing the minimum number of samples required to form a dense region.
    """
    # Iterate over each neighbor of the current sample
    for j in neighbors:
        if not visited[j]:
            visited[j] = True

            # Find all samples within eps distance of the current neighbor
            neighbors_j = find_neighbors(X, j, eps)

            # If the number of neighbors is greater than or equal to min_samples

