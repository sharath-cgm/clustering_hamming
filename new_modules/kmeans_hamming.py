import random
import numpy as np
from cluster_labelling import labelling
from sklearn.metrics.pairwise import manhattan_distances
from seeding import seeding1, seeding2
from lloyd_substitute import lloyd_substitute

class Kmeans:
	"""
	methods: 
	1) __init__
	2) _init_centroids (for seeding)
	3) fit (Lloyds algo)
	"""
	def __init__(
		self,
		n_clusters = 2,
		init = "random",
		n_init = 50,
		max_iter = 300,
		tol = 1e-4,
		# algorithm = "lloyd",
		# random_state = None,
		accuracy_list = []
	):
		self.n_clusters = n_clusters
		self.init = init
		self.n_init = n_init
		self.max_iter = max_iter
		self.tol = tol
		# self.algorithm = algorithm
		# self.random_state = random_state
		self.accuracy_list = accuracy_list

	def _init_centroids(self, X, x_squared_norms, init):
		"""
			return centers
		"""
		n_samples = X.shape[0]
		n_clusters = self.n_clusters


		if init == "seeding1":
			centers = seeding1(
				X,
				n_clusters
				# random_state = random_state,
				# x_squared_norms = x_squared_norms,
			)
		elif init == "seeding2":
			centers = seeding1(X, n_clusters)
		elif init == "random":
			seeds = np.random.permutation(n_samples)[:n_clusters]
			centers = X[seeds]

		return centers


	def fit(self, X, true_labels = None):
		"""
		X: data
		true_labels: Actual labels of data from the dataset; to get the epoch with max. accuracy against the generic max. inertia

		Returns:
		centers, labels
		"""

		n_samples = X.shape[0]

		# subtract of mean of X for more accurate distance computations
		# X_mean = X.mean(axis=0)
		# X -= X_mean

		# precompute squared norms of data points
		# x_squared_norms = np.reshape((X ** 2).sum(axis=1), (n_samples, 1))

		best_accuracy, best_inertia, best_labels = None, None, None

		for i in range(self.n_init):
			# initialize centers
			centers_init = self._init_centroids(
				X, x_squared_norms = x_squared_norms, init = self.init
			)

			# run lloyd's algo
			# labels, inertia, centers, n_iter_ = lloyds(X, centers_init,	x_squared_norms)
			labels, inertia, centers = lloyd_substitute(X, centers_init)
				# max_iter = self.max_iter,
				# tol = self.tol
				# x_squared_norms = 
			# )

			# compute accuracy

			if true_labels is not None:
				## best_accuracy
				accuracy = labelling(labels, true_labels, self.n_clusters, only_accuracy = True)
				(self.accuracy_list).append(accuracy)
				if best_accuracy is None or (accuracy > best_accuracy):
					best_labels = labels
					best_centers = centers
					best_accuracy = accuracy
					# best_n_iter = n_iter_
			else:
				## best_inertia
				if best_inertia is None or (inertia < best_inertia): # and same_clustering
					best_labels = labels
					best_centers = centers
					best_inertia = inertia
					# best_n_iter = n_iter_

		# ***** selfcopy*****
		# X += X_mean
		# best_centers += X_mean


		# returning
		self.cluster_centers_ = best_centers
		self.labels_ = best_labels
		self.inertia_ = best_inertia
		# self.n_iter_ = best_n_iter

		return self