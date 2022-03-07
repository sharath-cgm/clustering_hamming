from seeding import seeding


def lloyd(data, max_iter = 300):


	for i in range(max_iter):

		# compute dist to all centers and label

		# recompute the centers
		coordinate_count = np.zeroes(size= (k, D))
		total_count = np.zeroes(size = k)
		for i in range(n_samples):
			coordinate_count[labels[i]] += data[i]
			total_count[labels[i]] += 1

		centers = np.zeroes(size = (k, D))
		for i in range(k):
			rand_flips = np.random.uniform(size = D)
			for j in range(D):
				if rand_flips[i] < coordinate_count[i][j]/total_count[i]:
					centers[i][j] = 1

	return centers, labels


class Kmeans:
	def __init__(
		self,
		n_clusters = 2,
		n_init = 10,
		max_iter = 300,
		tol = 1e-4,
		init = "random"
		# algorithm = "lloyd",
		# random_state = None
	):
		self.n_clusters = n_clusters
		self.n_init = n_init
		self.max_iter = max_iter
		self.tol = tol
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

	# def predict

	def test_temp(self):
		print(self.n_clusters)

# k = Kmeans()
# k.test_temp()

class Temp:
	def hello(self):
		print("hello")