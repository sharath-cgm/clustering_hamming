import random
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


# def kmeans_plusplus: seeding method
def kmeans_plusplus(X, n_clusters, x_squared_norms, random_state = None, norm = 2):
	"""
 	X : data

	n_clusters : number of seeds to choose

	norm : default 2, to generalize for l-p norm
 	---------

 	Returns centers
 	"""
  
	if random_state is not None:
 		np.random.seed(random_state)
  
	n_samples, n_features = X.shape
	centers = np.empty((n_clusters, n_features), dtype=X.dtype)
	n_local_trials = 2 + int(np.log(n_clusters))

	# precompute
	# x_squared_norms = (X * X).sum(axis=1) ### 1

	# Pick first center randomly
	center_id = np.random.randint(0, n_samples-1) ##2
	centers[0] = X[center_id]

	# Initialize list of closest distances and calculate current potential
	closest_dist_sq = euclidean_distances(
		centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True
	)
	current_pot = closest_dist_sq.sum()
	# print(closest_dist_sq, type(closest_dist_sq))


	# Pick the remaining n_clusters-1 points
	for c in range(1, n_clusters):
		# Choose center candidates by sampling with probability proportional to the squared distance to the closest existing center
		if norm == 2:
			probability = closest_dist_sq
		else:
			probability = np.sqrt(closest_dist_sq)
			# print(type(probability))
			if(norm > 2):
				probability = np.power(probability, norm)
		# probability = closest_dist_sq

		rand_vals = np.random.uniform(size=n_local_trials) * probability.sum()
		candidate_ids = np.searchsorted(np.cumsum(probability, dtype=np.float64), rand_vals) ##3
		np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)  ##4

		# Compute distances to center candidates
		distance_to_candidates = euclidean_distances(
		  X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True
		)

		# update closest distances squared and potential for each candidate
		np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
		candidates_pot = distance_to_candidates.sum(axis=1)

		# Decide which candidate is the best
		best_candidate = np.argmin(candidates_pot)
		current_pot = candidates_pot[best_candidate]
		closest_dist_sq = distance_to_candidates[best_candidate]
		best_candidate = candidate_ids[best_candidate]

		# Add best center candidate found in local tries
		centers[c] = X[best_candidate]

	return centers


# def lloyd_algo
def lloyd_iter(X, x_squared_norms, centers_old, centers_new, labels, center_shift):
	
	
    # distance between data points and centers
    pairwise_dist = nd.array

    for i in range(n_samples):
        for j in range(n_samples):
            pairwise_dist[i][j] = 1 # todo

    # label data points
    for i in range(n_samples):
        min_sq_dist = pairwise_dist[i][0]
        label = 0
        for j in range(1, n_clusters):
            sq_dist = pairwise_dist[i][j]
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                label = j
        labels[i] = label

    # update centers

    return 0

def lloyds(X, centers_init, x_squared_norms, max_iter = 300, tol=1e-4):
    """
    inputs:
    X : data
    centers_init : initial centers
    x_squared_norms : Precomputed x_squared_norms
    max_iter : Maximum number of iterations, default = 300
    tol : default= 1e-4, for convergence with small relative change of centers
    --------------------
    outputs:
    centers : in the last iteration
    label : label[i] index of centroid the i_th observation is closest to

    """

    n_clusters = centers_init.shape[0]

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init
    centers_new = np.zeros_like(centers)
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    # weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)

    # strict_convergence = False

    # iterate
    for i in range(max_iter):
        lloyd_iter() # todo, no need for function

        centers, centers_new = centers_new, centers

        if np.array_equal(labels, labels_old):
            # strict_convergence = True
            break
        else:
            center_shift_tot = (center_shift ** 2).sum()
            if center_shift_tot <= tol:
                break

        labels_old[:] = labels

    # if not strict_convergence:
    #   lloyd_iter()

    return labels, centers


class Kmeans:
	def __init__(
		self,
		n_clusters = 2,
		init = "random",
		n_init = 10,
		max_iter = 300,
		tol = 1e-4,
		# algorithm = "lloyd",
		random_state = None
	):
		self.n_clusters = n_clusters
		self.init = init
		self.n_init = n_init
		self.max_iter = max_iter
		self.tol = tol
		# self.algorithm = algorithm
		self.random_state = random_state

	def _init_centroids(self, X, x_squared_norms, init, random_state):
		"""
			return centers
		"""
		n_samples = X.shape[0]
		n_clusters = self.n_clusters


		if init == "k-means++":
			centers = kmeans_plusplus(
				X,
				n_clusters,
				random_state = random_state,
				x_squared_norms = x_squared_norms,
			)
		elif init == "random":
			seeds = random_state.permutation(n_samples)[:n_clusters]
			centers = X[seeds]

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