# from seeding import seeding
"""
todo: manhattan distance is expensive, XOR and count number of 1s maybe is efficient
"""
from sklearn.metrics.pairwise import manhattan_distances


# def lloyds(X, centers_init, x_squared_norms, max_iter = 300, tol=1e-4):
def lloyd_substitute(X, centers_init, max_iter = 300):
	n_samples, n_features = X.shape
	n_clusters = centers_init.shape[0]

	# Buffers to avoid new allocations at each iteration.
	centers = centers_init
	# centers_new = np.zeros_like(centers)
	labels = np.full(X.shape[0], -1, dtype=np.int32)
	labels_old = labels.copy()
	# count_in_clusters = np.zeros(n_clusters, dtype=X.dtype) # count of samples in each cluster
	# center_shift = np.zeros(n_clusters, dtype=X.dtype)
	pairwise_dist = np.zeros((n_samples, n_clusters), dtype=X.dtype)

	# iterate- lloyds algorithm
	for _ in range(max_iter):
		# compute dist to all centers
		pairwise_dist = manhattan_distances(X = X, Y = centers)

		# label data points and update centers
		coordinate_count = np.zeroes(size= (n_clusters, n_features))
		total_count = np.zeroes(size = n_clusters)

		for j in range(n_samples):
			labels[j] = np.argmin(pairwise_dist[j])

			coordinate_count[labels[j]] += data[j]
			total_count[labels[j]] += 1

		# centers_new = np.zeros_like(centers) 	# without center_shift, centers_new isnt necessary

		centers = np.zeroes(size = (n_clusters, n_features))
		for j in range(n_clusters):
			rand_flips = np.random.uniform(size = n_features)
			for k in range(n_features):
				if rand_flips[k] < coordinate_count[j][k]/total_count[j]:
					centers[j][k] = 1

		if np.array_equal(labels, labels_old): # convergence condition #2 : when labels don't change
			# strict_convergence = True
			break

		labels_old[:] = labels

	# compute inertia
	inertia = 0.0
	for j in range(n_samples):
		inertia += pairwise_dist[j][labels[j]]**2

	return labels, inertia, centers


class Kmeans:
	def __init__(
		self,
		n_clusters = 2,
		n_init = 10,
		max_iter = 300,
		# tol = 1e-4,
		init = "random"
		# algorithm = "lloyd",
		# random_state = None
	):
		self.n_clusters = n_clusters
		self.n_init = n_init
		self.max_iter = max_iter
		# self.tol = tol
		self.init = init
		# self.algorithm = algorithm
		# self.random_state = random_state

	def _init_centroids(self, X, x_squared_norms):
		"""
			return centers
		"""
		n_samples = X.shape[0]
		n_clusters = self.n_clusters


		if init == "random":
			seeds = random_state.permutation(n_samples)[:n_clusters]
			centers = X[seeds]
		else:
			centers = seeding(
				X,
				n_clusters
				# random_state = random_state,
				# x_squared_norms = x_squared_norms,
			)

		return centers

	def fit(self, X):
		"""
		description
		"""

		# subtract of mean of X for more accurate distance computations
		X_mean = X.mean(axis=0)
		X -= X_mean

		# precompute squared norms of data points
		x_squared_norms = (X * X).sum(axis=1)

		best_inertia, best_labels = None, None

		for i in range(self.n_init):
			# initialize centers
			centers_init = self._init_centroids(
				X, x_squared_norms = x_squared_norms, init = init, random_state = random_state
			)

			# run lloyd's algo
			labels, inertia, centers, n_iter_ = lloyd(
				X,
				centers_init,
				max_iter = self.max_iter,
				tol = self.tol,
				x_squared_norms = x_squared_norms,
			)

			## best_accuracy

			if best_inertia is None or (inertia < best_inertia): # and same_clustering
				best_labels = labels
				best_centers = centers
				best_inertia = inertia
				best_n_iter = n_iter_

		# ***** selfcopy*****
		X += X_mean
		best_centers += X_mean


		# returning
		self.cluster_centers_ = best_centers
		self.labels_ = best_labels
		self.inertia_ = best_inertia
		self.n_iter_ = best_n_iter

		return self
