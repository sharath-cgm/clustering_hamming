from sklearn.metrics.pairwise import manhattan_distances
import numpy as np

def quality_data(centers):
	# measure min hamming distance between all pairs of centers

	n_centers, min_dist = centers.shape

	for i in range(n_centers):
		for j in range(i+1, n_centers):
			dist = manhattan_distances([centers[i]],[centers[j]])[0][0]
			if min_dist > dist:
				min_dist = dist

	return min_dist

def flip(center, alpha):
	D = center.shape[0]
	rand_flips = np.random.uniform(size = D)
	new = np.copy(center)

	for i in range(D):
		if rand_flips[i] < alpha:
			new[i] = 1 - new[i]
	return new

def generation(N, D, k, alpha = 0.2):
	"""
	Input:
	N : number of samples to generated
	D : dimension
	k : number of distinct clusters
	alpha : cluster clarity (measure of how compact each cluster is)
	------------------------------------
	returns-
	data: (N * D) array
	centers: (k * D) array
	labels: (D*1) array
	"""

	# randomly choose k D-dimensional centers
	centers= np.random.randint(2, size= (k, D))
	labels = list(range(k))

	data = np.array(centers, dtype = int)
	# Add N/k - 1 points into each cluster
	n_samples = int(N/k - 1)
	for i in range(k):
		# print(i)
		for j in range(n_samples):
			# print(j)
			# new = centers[i] flip into prob. alpha
			# print(j)
			new = flip(centers[i], alpha)
			# print(new)
			data = np.append(data, [new], axis=0)

			# check if already present in data
			labels.append(i)

	labels = (np.array([labels])).transpose()

	return data, centers, labels

# data, centers, labels = generation(N = 100000, D = 1000, k = 10000)
# data, centers, labels = generation(N = 100000, D = 1000, k = 2)
# data, centers, labels = generation(N = 500000, D = 50, k = 2)
# data, centers, labels = generation(N = 500000, D = 50, k = 20000)
data, centers, labels = generation(N = 100000, D = 100, k = 20)


# print(type(data[0][0]))
# data, centers, labels = generation(N = 50000, D = 50, k = 100)
# print(data.shape)
# print("\n\n",centers)
# np.savetxt("N_3000_2.txt", np.append(data, labels, axis = 1), delimiter=" ")
print("writing")
np.savetxt("N_100000_1.txt", np.append(data, labels, axis = 1), delimiter=" ")



unique_entries, _ = (np.unique(data, axis=0)).shape
print(unique_entries)

print(quality_data(centers))
