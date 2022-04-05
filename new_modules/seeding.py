"""
todo:

"""

import numpy as np
from sklearn.metrics.pairwise import manhattan_distances

def max_ones(data, chosen_centers):
	"""
	chosen_centers: these are indices of the centers
	"""

	n_samples, _= data.shape
	l, max = [], -1

	for i in range(n_samples):
		count = np.count_nonzero(data[i])
		if count > max:
			l = []
			l.append(i)
			max = count
		elif count == max:
			l.append(i)

	# delete chosen_centers from l
	l2 = set(chosen_centers)
	l = [x for x in l if x not in l2]

	return np.random.choice(l)


def seeding1(data, n_clusters):
	"""
	Implementation based on approximation algorithm of a set cover

	data : data in hamming metric
	n_clusters : number of seeds to choose, 'k'

	---------

	Returns:
	centers: indices of centers
	"""

	n_samples, n_features = data.shape
	# centers = np.empty(n_clusters, dtype= data.dtype)
	centers = [-1] * n_clusters

	# Pick first center randomly
	centers[0] = np.random.randint(0, n_samples)
	center = data[centers[0]].copy()
	# centers[0] = data[center_id]

	# Perform XOR operation 
	for i in range(n_samples):
		data[i] = np.bitwise_xor(data[i], center)
		# print(first_center)

	# print(data)

	# choose the rest of (k-1) centers
	for i in range(1, n_clusters):
		new_center = max_ones(data, centers)
		centers[i] = new_center
		center = data[new_center].copy()

		for j in range(n_samples):
			data[j] = np.bitwise_xor(data[j], center, out = data[j], where = data[j].astype(dtype = bool))

	# get centers from indices

	return centers


def seeding2(data, n_clusters):
	"""
	implementation similar to k-means++, choose farthest points
	"""
	n_samples, n_features = data.shape
	centers = np.empty((n_clusters, n_features), dtype= data.dtype)

	# Pick first center randomly
	centers[0] = data[np.random.randint(0, n_samples)]

	# Initialize list of closest distances
	closest_dist = manhattan_distances(centers[0, np.newaxis], data)

	# Pick the remaining n_clusters-1 points
	for c in range(1, n_clusters):
		print(closest_dist)
		# Choose center candidate by finding the farthest to existing center
		center_index = np.argmax(closest_dist)
		centers[c] = data[center_index]

		# Compute distances to center candidates
		new_dist = manhattan_distances(data[center_index, np.newaxis], data)

		# update closest distances
		np.minimum(closest_dist, new_dist, out=closest_dist)

	return centers

data = np.array([ [0, 0, 0, 0, 0, 0], [1,1, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1],  [0, 0, 0, 0, 0, 1], [1,1,1,0,0,0], [1,1,1,1,0,0] ])

print("\ncenters:")
print(seeding2(data, 3))
# print(data)
# print(max_ones(data))