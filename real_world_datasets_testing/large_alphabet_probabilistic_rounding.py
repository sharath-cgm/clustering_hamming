import numpy as np
from cluster_labelling import labelling
from sklearn.metrics import accuracy_score
from distance_measures import hamming_distance
import math

def tanh(x, a, t):
	exp = np.exp(t*a * (x - 1/a))
	return (1 + (a-1)*(exp-1)/(exp + a - 1))/a

def probabilistic_rounding_large_alphabets(X, centers_init, max_iter, tanh_t, t, number_discrete_values_in_features, true_labels = None):
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


	# to replace fraction_of_ones
	count_alphabets = np.zeros(shape= (n_clusters, n_features, max(number_discrete_values_in_features)))


	# iterate- lloyds algorithm
	for iter in range(max_iter):
		# kmode_score_current = 0

		# print("\niter", iter)
		# compute dist to all centers
		# print(X.shape, centers.shape)
		# pairwise_distance = euclidean_distances(X = X, Y = center_old, squared = True) # Squared euclidean gives similar comparison as hamming distance
		pairwise_distance = hamming_distance(X = X, Y = center_old)
		# pairwise_dist = hamming_distance(X = X, Y = centers)


		# fraction_of_ones = np.zeros(shape= (n_clusters, n_features))
		count_alphabets = np.zeros(shape= (n_clusters, n_features, max(number_discrete_values_in_features)))

		## label data points
		#1 choose center randomly in case two centers are equidistant
		min_distance = np.min(pairwise_distance, axis = 1)
		for j in range(n_samples):
			minimum_indices = np.where(pairwise_distance[j] == min_distance[j])[0]
			labels[j] = np.random.choice(minimum_indices)

			for k in range(n_features):
				count_alphabets[labels[j]][k][X[j][k]] += 1


		## update centers
		_, total_count = np.unique(labels, return_counts = True)
		# fraction_of_ones /= total_count.reshape(-1,1)
		for j in range(n_clusters):
			count_alphabets[j] /= total_count[j] # fraction

		# print(count_alphabets)

		# probabilistic rounding
		# tanh_t = 10
		# rand_flips = np.random.uniform(low = -1, high = 1, size = (n_clusters, n_features))
		
		for j in range(n_clusters):
			# print("cluster", j)
			for k in range(n_features):
				# print("feature", k)

				l = number_discrete_values_in_features[k]
				z = count_alphabets[j][k]
				# distance_from_simplex_vertices = np.zeros(shape = l)
				# for m in range(l):
				# 	z[m] -= 0.707106
				# 	distance_from_simplex_vertices[m] = np.linalg.norm(z)
				# 	z[m] += 0.707106

				# print("p",p/1.4142)
				# p = np.tanh(tanh_t * p)
				# p = tanh(p, l, tanh_t)

				# total = np.sum(distance_from_simplex_vertices)
				# if total != 0:
				# 	distance_from_simplex_vertices /= total

				probability = np.zeros(shape = l)
				probability= (z[0:l])**t

				# for i in range(l):
					# probability[i] = tanh(distance_from_simplex_vertices[i], l, tanh_t)
					# probability[i] = tanh(z[i], l, tanh_t)

					# probability[i] = z[i]**t
					# print(probability)

				# probability = (z[0:l])**t

				# print("tanh p",p, "\n")
				total = np.sum(probability)
				if total != 0:
					normalize_p = probability/total
				else:
					normalize_p = np.array([1])

				# print(normalize_p)
				centers[j][k] = np.random.choice(list(range(l)), p = normalize_p)
				# centers[j][k] = np.random.choice(list(range(l)), p = z)


		# for j in range(n_clusters):
		# 	for k in range(n_features):
		# 		max_count = np.max(count_alphabets[j][k])
		# 		max_indices = np.where(count_alphabets[j][k] == max_count)[0]
		# 		centers[j][k] = np.random.choice(max_indices)

				# print(max_count, centers[j][k])


		# label_changes.append(np.count_nonzero(labels != labels_old))
		if np.array_equal(labels, labels_old): # convergence condition #2 : when labels don't change
			# strict_convergence = True
			break

		center_old[:] = centers
		labels_old[:] = labels

		# print(centers)


		if true_labels is not None:
			predicted_labels = labelling(labels, true_labels, n_clusters, n_samples)
			accuracy.append(accuracy_score(true_labels, predicted_labels))
			# accuracy.append([accuracy_score(true_labels, predicted_labels), kmode_score_current])


	# majority rounding
	# centers = fraction_of_ones > 0.5
	# centers.astype(int)

	print("Accuracies over the iterations: ", accuracy)

	print(" ")

	return labels, centers

def majority_rounding_large_alphabets(X, centers_init, max_iter, number_discrete_values_in_features, true_labels = None):
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


	# to replace fraction_of_ones
	count_alphabets = np.zeros(shape= (n_clusters, n_features, max(number_discrete_values_in_features)))


	# iterate- lloyds algorithm
	for iter in range(max_iter):
		# kmode_score_current = 0

		# print(iter, "iter")
		# compute dist to all centers
		# print(X.shape, centers.shape)
		# pairwise_distance = euclidean_distances(X = X, Y = center_old, squared = True) # Squared euclidean gives similar comparison as hamming distance
		pairwise_distance = hamming_distance(X = X, Y = center_old)
		# pairwise_dist = hamming_distance(X = X, Y = centers)


		# fraction_of_ones = np.zeros(shape= (n_clusters, n_features))
		count_alphabets = np.zeros(shape= (n_clusters, n_features, max(number_discrete_values_in_features)))

		## label data points
		#1 choose center randomly in case two centers are equidistant
		min_distance = np.min(pairwise_distance, axis = 1)
		for j in range(n_samples):
			minimum_indices = np.where(pairwise_distance[j] == min_distance[j])[0]
			labels[j] = np.random.choice(minimum_indices)

			for k in range(n_features):
				count_alphabets[labels[j]][k][X[j][k]] += 1


		## update centers
		_, total_count = np.unique(labels, return_counts = True)
		# fraction_of_ones /= total_count.reshape(-1,1)
		# for j in range(n_clusters):
		# 	count_alphabets[j] /= total_count[j]


		# probabilistic rounding
		# tanh_t = 10
		# rand_flips = np.random.uniform(low = -1, high = 1, size = (n_clusters, n_features))
		
		# for j in range(n_clusters):
		# 	for k in range(n_features):
		# 		l = number_discrete_values_in_features[k]
		# 		p = np.zeros(shape = l)
		# 		z = np.copy(count_alphabets[j][k])
		# 		for m in range(l):
		# 			z[m] -= 1
		# 			p[m] = np.linalg.norm(z)
		# 			z[m] += 1

		# 		p = np.tanh(tanh_t* p)
		# 		normalize = np.sum(p)
		# 		centers[j][k] = np.random.choice(list(range(l)), p = p/normalize)

		for j in range(n_clusters):
			for k in range(n_features):
				max_count = np.max(count_alphabets[j][k])
				max_indices = np.where(count_alphabets[j][k] == max_count)[0]
				centers[j][k] = np.random.choice(max_indices)
				
				# print(max_count, centers[j][k])


		# label_changes.append(np.count_nonzero(labels != labels_old))
		if np.array_equal(labels, labels_old): # convergence condition #2 : when labels don't change
			# strict_convergence = True
			break

		center_old[:] = centers
		labels_old[:] = labels

		# print(centers)


		if true_labels is not None:
			predicted_labels = labelling(labels, true_labels, n_clusters, n_samples)
			accuracy.append(accuracy_score(true_labels, predicted_labels))
			# accuracy.append([accuracy_score(true_labels, predicted_labels), kmode_score_current])


	# majority rounding
	# centers = fraction_of_ones > 0.5
	# centers.astype(int)

	print("Accuracies over the iterations: ", accuracy)

	print(" ")

	return labels, centers