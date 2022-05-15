"""
Various distance that are used in this project are defined here

1) Euclidean distance
2) Hamming distance
3) Manhattan distance
4) Generic lp-norm

Input: two vectors
returns corresponding distance between them in corresponding metric
"""

import numpy as np

# def euclidean_distance(x, y):

def hamming_distance(X, Y):
	n_samples, n_features = X.shape
	n_clusters, _ = Y.shape

	# initialize
	pairwise_distance = np.ndarray( shape= (n_samples, n_clusters), dtype= int)

	for i in range(n_samples):
		for j in range(n_clusters):
			pairwise_distance[i][j] = np.count_nonzero(X[i] != Y[j])
	
	return pairwise_distance
