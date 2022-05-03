"""
todo: manhattan distance is expensive, XOR and count number of 1s maybe is efficient
"""
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances

import numpy as np
from cluster_labelling import labelling
from sklearn.metrics import accuracy_score
from distance_measures import hamming_distance

def center_shift(centers, old_centers):
	# k, D = centers.shape
	# xor = np.bitwise_xor(centers, old_centers)
	# return np.amax(np.count_nonzero(xor, axis = 1))
	xor = centers != old_centers
	return np.count_nonzero(xor, axis = 1)


# def lloyds(X, centers_init, x_squared_norms, max_iter = 300, tol=1e-4):
def probabilistic_rounding(X, centers_init, max_iter, tanh_t, true_labels = None):
	n_samples, n_features = X.shape
	# print(type(np.array(centers_init)))
	n_clusters = centers_init.shape[0]

	# Buffers to avoid new allocations at each iteration.
	centers = centers_init
	# centers_new = np.zeros_like(centers)
	center_old = centers.copy()
	labels = np.full(X.shape[0], -1, dtype=np.int32)
	labels_old = labels.copy()
	# count_in_clusters = np.zeros(n_clusters, dtype=X.dtype) # count of samples in each cluster
	# center_shift = np.zeros(n_clusters, dtype=X.dtype)
	pairwise_dist = np.zeros((n_samples, n_clusters), dtype=X.dtype)
	coordinate_count = np.zeros(shape= (n_clusters, n_features))
	total_count = np.zeros(shape = n_clusters)

	accuracy = []
	label_changes = []
	centers_shift = []

	# iterate- lloyds algorithm
	for iter in range(max_iter):
		# print(iter, "iter")
		# compute dist to all centers
		# print(X.shape, centers.shape)
		pairwise_distance = euclidean_distances(X = X, Y = centers, squared = True) # Squared euclidean gives similar comparison as hamming distance
		# pairwise_dist = hamming_distance(X = X, Y = centers)


		fraction_of_ones = np.zeros(shape= (n_clusters, n_features))

		## label data points
		#1 choose center randomly in case two centers are equidistant
		min_distance = np.min(pairwise_distance, axis = 1)
		for j in range(n_samples):
			minimum_indices = np.where(pairwise_distance[j] == min_distance[j])[0]
			labels[j] = np.random.choice(minimum_indices)

			fraction_of_ones[labels[j]] += X[j]
		
		#2 easier implementation: choose the closest center using argmin which return the first occurence of min value
		# labels = np.argmin(pairwise_distance, axis = 1)
		# print(labels.shape)

		## update centers
		_, total_count = np.unique(labels, return_counts = True)
		fraction_of_ones /= total_count.reshape(-1,1)

		# probabilistic rounding
		# tanh_t = 10
		rand_flips = np.random.uniform(low = -1, high = 1, size = (n_clusters, n_features))
		centers = rand_flips < np.tanh( tanh_t* (fraction_of_ones - 0.5))
		centers.astype(int)

		# majority rounding
		# centers = fraction_of_ones > 0.5
		# centers.astype(int)





		# # label data points and update centers
		# coordinate_count = np.zeros(shape= (n_clusters, n_features))
		# total_count = np.zeros(shape = n_clusters)

		# # TODO-  use numpy functions
		# for j in range(n_samples):
		# 	# labels[j] = np.argmin(pairwise_dist[j])
		# 	min = np.min(pairwise_dist[j])
		# 	minimum_indices = np.where(pairwise_dist[j] == min)[0]
		# 	labels[j] = np.random.choice(minimum_indices)

		# 	coordinate_count[labels[j]] += X[j]
		# 	total_count[labels[j]] += 1

		# # centers_new = np.zeros_like(centers) 	# without center_shift, centers_new isnt necessary
		# # print(labels)
		# centers = np.zeros(shape = (n_clusters, n_features))
		
		# t = 10 # hyperparameter in Tanh function, tried t = 5, 10, 12, 20

		# for j in range(n_clusters):

		# 	## TODO: use numpy functions
		# 	rand_flips = np.random.uniform(size = n_features)
		# 	for k in range(n_features):
		# 		# probabilistic approach using Tanh function
		# 		p = coordinate_count[j][k]/ total_count[j]

		# 		if rand_flips[k] < 0.5*(np.tanh(t*(p-0.5)) + 1):
		# 			centers[j][k] = 1

				# simple majority based approach
				# if coordinate_count[j][k] > total_count[j] * 0.5: 
				# 	centers[j][k] = 1
					
				# trial 1
				# p = coordinate_count[j][k]/ total_count[j]
				# if p > 0.5:
				# 	if rand_flips[k] < 1 - 8 * (1-p)**4:
				# 		centers[j][k] = 1
				# else:
				# 	if rand_flips[k] < 8 * p**4:
				# 		centers[j][k] = 1


		# label_changes.append(np.count_nonzero(labels != labels_old))
		if np.array_equal(labels, labels_old): # convergence condition #2 : when labels don't change
			# strict_convergence = True
			break

		########## TODO
		# else if center_shift(centers, old_centers) <= 2:
			# perform simple majority
			# for j in range(n_clusters):

		# centers_shift.append(center_shift(centers, center_old))

		center_old[:] = centers
		labels_old[:] = labels



		if true_labels is not None:
			predicted_labels = labelling(labels, true_labels, n_clusters, n_samples)
			accuracy.append(accuracy_score(true_labels, predicted_labels))

		# unique, counts = np.unique(labels, return_counts=True)
		# label_changes.append(dict(zip(unique, counts)))



	# compute inertia
	# inertia = 0.0
	# for j in range(n_samples):
	# 	inertia += pairwise_dist[j][labels[j]]**2

	# return labels, inertia, centers, accuracy
	# print("Number of Label changes between previous and current iteration: ", label_changes)

	# print("Number of samples per cluster: \n", np.array(label_changes))
	# print("Hamming distance between centers of current and previous iteration: \n", np.array(centers_shift))
	print("Accuracies over the iterations: ", accuracy)
	print(" ")

	return labels, centers
