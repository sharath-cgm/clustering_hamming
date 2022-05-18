import random
import numpy as np
from cluster_labelling import labelling
from sklearn.metrics.pairwise import manhattan_distances
from seeding import kmeans_plusplus
# from lloyd_substitute import lloyd_substitute
# from probabilistic_rounding import probabilistic_rounding, majority_rounding
from large_alphabet_probabilistic_rounding import probabilistic_rounding_large_alphabets, majority_rounding_large_alphabets
from sklearn.metrics import accuracy_score, f1_score

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
		n_init = 10,
		max_iter = 50,
		tanh_t = 7,
		algorithm = "probabilistic_rounding",
		t = 2
		# tol = 1e-4,
		# seed = None,
		# random_state = None,
		# accuracy_list = [],
		# f1_list = []
	):
		self.n_clusters = n_clusters
		self.init = init
		self.n_init = n_init
		self.max_iter = max_iter
		self.tanh_t = tanh_t
		self.algorithm = algorithm
		self.t = t
		# self.tol = tol
		# self.random_state = random_state
		# self.seed = seed,
		# self.accuracy_list = list(accuracy_list),
		# self.f1_list = f1_list

	def _init_centroids(self, X, init):
		"""
			return centers
		"""
		n_samples = X.shape[0]
		n_clusters = self.n_clusters

		if init == "k-means++":
			centers = kmeans_plusplus(X, self.n_clusters)
		elif init == "random":
			seeds = np.random.permutation(n_samples)[:n_clusters]
			centers = X[seeds]

		return centers


	def fit(self, X, number_discrete_values_in_features = None, true_labels = None, input_seed = None):
		"""
		X: data
		true_labels: Actual labels of data from the dataset; to get the epoch with max. accuracy against the generic max. inertia
		input_seed: Hard-code the 

		Returns:
		centers, labels
		"""

		n_samples, n_features = X.shape

		self.accuracy_list = []
		self.f1_list = []

		# best_accuracy, best_f1_score, best_labels = None, None, None

		for i in range(self.n_init):
			# print("epoch number: ", i)

			# initialize centers
			if input_seed == None:
				centers_init = self._init_centroids(X, init = self.init)
			else:
				centers_init = input_seed


			# run lloyd's algo
			# labels, inertia, centers = lloyd_substitute(X, centers_init, self.max_iter, true_labels)
			# labels, centers, accuracy_iter = lloyd_substitute(X, centers_init, self.max_iter, true_labels)
			# labels, centers = lloyd_substitute(X, centers_init, self.max_iter, true_labels)
			
			if self.algorithm == "probabilistic_rounding":
				labels, centers = probabilistic_rounding(X, centers_init, self.max_iter, self.tanh_t, self.t, true_labels)
			elif self.algorithm == "majority_rounding":
				labels, centers = majority_rounding(X, centers_init, self.max_iter, true_labels)
			elif self.algorithm == "probabilistic_rounding_large_alphabets":
				# number_discrete_values_in_features = np.zeros(shape = n_features, dtype = int)
				# for j in range(n_features):
				# 	number_discrete_values_in_features[j] = len(np.unique(X[:, j]))
				# print(number_discrete_values_in_features)
				labels, centers = probabilistic_rounding_large_alphabets(X, centers_init, self.max_iter, self.tanh_t, self.t, number_discrete_values_in_features, true_labels)
			elif self.algorithm == "majority_rounding_large_alphabets":
				# number_discrete_values_in_features = np.zeros(shape = n_features, dtype = int)
				# for j in range(n_features):
				# 	number_discrete_values_in_features[j] = len(np.unique(X[:, j]))
				# print(number_discrete_values_in_features)
				labels, centers = majority_rounding_large_alphabets(X, centers_init, self.max_iter, number_discrete_values_in_features, true_labels)

			# self.accuracy = accuracy_iter
			# print(accuracy_iter, len(accuracy_iter))


			# compute accuracy
			if true_labels is not None:
				## best_accuracy
				predicted_labels = labelling(labels, true_labels, self.n_clusters, n_samples)
				(self.accuracy_list).append(accuracy_score(labels, true_labels))
				(self.f1_list).append(f1_score(labels, true_labels, average='macro'))
				# if best_accuracy is None or (accuracy > best_accuracy):
				# 	best_labels = labels
				# 	best_centers = centers
				# 	best_accuracy = accuracy
					# best_n_iter = n_iter_
			# else:
			# 	## best_inertia
			# 	if best_inertia is None or (inertia < best_inertia): # and same_clustering
			# 		best_labels = labels
			# 		best_centers = centers
			# 		best_inertia = inertia
					# best_n_iter = n_iter_

		# returning
		# self.cluster_centers_ = best_centers
		# self.labels_ = best_labels
		# self.inertia_ = best_inertia
		# self.n_iter_ = best_n_iter


		return self